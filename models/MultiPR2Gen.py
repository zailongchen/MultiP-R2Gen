import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import CLIPModel, MistralForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
# from medclip import MedCLIPModel, MedCLIPVisionModelViT
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.distributed as dist


class MLP(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim):
        super(MLP,self).__init__()
        self.hidden_1 = nn.Linear(in_dim, inter_dim)
        self.act = nn.ReLU()
        self.hidden_2 = nn.Linear(inter_dim, out_dim)
    
    def forward(self,x):
        a = self.act(self.hidden_1(x))
        return self.hidden_2(a)

class R2GenGPT(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading llama')
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        # self.llama_tokenizer.pad_token_id = 0
        # if args.low_resource:
        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #         args.llama_model,
        #         torch_dtype=torch.float16,
        #         load_in_8bit=True,
        #         device_map="auto"
        #     )
        # else:
        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #         args.llama_model,
        #         torch_dtype=torch.bfloat16,
        #     )

        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.pmc_llama, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0  # for llama2-7b and pmc-llama
        # self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id # for mmed_llama3
        print(self.llama_tokenizer.bos_token_id)
        print(self.llama_tokenizer.eos_token_id)
        print(self.llama_tokenizer.pad_token_id)
        if args.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.pmc_llama,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.pmc_llama,
                torch_dtype=torch.float16,
            )


        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')
        
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size) # 768 for medclip

        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym  # for llama2-7b and pmc-llama
        # self.end_sym = "<|end_of_text|>"  # for mmed-llama3

        if self.args.task == 'label':
            self.prompt = "<<SYS>>\nYou are a medical assistant, you will generate a status for each of the 14 disease labels, indicating the key clinical information depicted in this chest xray image. The 14 disease labels are as follows: Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, No Finding. Please output \"present\" or \"absent\" or \"uncertain\" or \"missing\" for each label. \"present\" indicates this label is present in the the corresponding image. \"absent\" indicates this label should not be present in the the corresponding image. \"uncertain\" indicates this label may or may not be present to some degree in the corresponding image. \"missing\" indicates this label is not about the corresponding image. \n<</SYS>>\n\nThe chest xray image is"
        elif self.args.task == 'triple':
            self.prompt = "<<SYS>>\nYou are a medical assistant, you will generate all possible Entity-Ralation triples indicating the key clinical information depicted in this chest xray image. Please use four types of triples: entity suggestive_of entity, entity located_at entity, entity modify entity , and entity status present. Suggestive_of is a relation between two observation entities indicating that the presence of the second observation is inferred from that of the ﬁrst observation. Located_at is a relation between an observation entity and an anatomy entity indicating that the observation is related to the anatomy. Modify is a relation between two observation entities or two anatomy entities indicating that the ﬁrst entity modiﬁes the scope of, or quantiﬁes the degree of, the second entity. Status present indicats that the entity is presented in the image. \n<</SYS>>\n\nThe chest xray image is"
        elif self.args.task == 'report':
            self.prompt = "<<SYS>>\nYou are a medical assistant, you will generate a comprehensive and detailed diagnosis report for this chest xray image. \n<</SYS>>\n\nThe chest xray image is"
        
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')
    

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            # (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            # swin transformer
            visual_outputs = self.visual_encoder(image)
            image_embed = visual_outputs['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)

        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'[INST] {self.prompt} <Img><ImageHere></Img> [/INST] '
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)  

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        all_loss = outputs.loss

        return {"loss": all_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}_ori.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        text = samples["text"]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=0,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split(self.end_sym)[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        self.val_step_outputs = self.all_gather(self.val_step_outputs)
        ref, hypo, ids = [], [], []
        for cnt in range(len(self.val_step_outputs)):
            for i in self.val_step_outputs[cnt]:
                ref.extend(i['ref'])
                hypo.extend(i['hypo'])
                ids.extend(i['id'])
        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        text = samples["text"]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=0,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        self.test_step_outputs = self.all_gather(self.test_step_outputs)
        ref, hypo, ids = [], [], []
        for cnt in range(len(self.test_step_outputs)):
            for i in self.test_step_outputs[cnt]:
                ref.extend(i['ref'])
                hypo.extend(i['hypo'])
                ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

    @torch.no_grad()
    def all_gather(self, data):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        dist.barrier()
        gather_data = [None for _ in range(torch.distributed.get_world_size())]
        dist.all_gather_object(gather_data, data)
        return gather_data
    

    def Clip_loss(self, image_embeds, text_embeds): # image_embeds: [6, 4096], text_embeds: [6, 4096]
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device).long()

        return (F.cross_entropy(logits_per_text, labels) +
                F.cross_entropy(logits_per_image, labels)) / 2

    def Temporal_weight(self, text_embeds, text_mask, mode='former'):
        bs, N = text_embeds.shape[0], text_embeds.shape[1]
        if mode=='former':
            for bs_idx in range(bs):
                one_sent = 0.0
                weights = 0.0
                num_tokens = sum(text_mask[bs_idx])
                for idx in range(num_tokens):
                    w = (num_tokens - idx + 1) ** (-0.5)
                    one_sent += w * text_embeds[bs_idx,idx,:]
                    weights += w
                one_sent = one_sent / weights
                if bs_idx == 0:
                    all_embeds = one_sent.unsqueeze(0)
                else:
                    all_embeds = torch.cat((all_embeds, one_sent.unsqueeze(0)), dim=0)

        elif mode=='latter':
            all_embeds = 0.0
            weights = 0.0
            for idx in range(N):
                w = (N - idx + 1) ** (-0.5)
                all_embeds += w * text_embeds[:,idx,:]
                weights += w
            all_embeds = all_embeds / weights
        
        elif mode=='mean':
            for bs_idx in range(bs):
                one_sent = 0.0
                cnt = 0.0
                num_tokens = sum(text_mask[bs_idx])
                for idx in range(num_tokens):
                    one_sent += text_embeds[bs_idx,idx,:]
                    cnt += 1
                one_sent = one_sent / cnt
                if bs_idx == 0:
                    all_embeds = one_sent.unsqueeze(0)
                else:
                    all_embeds = torch.cat((all_embeds, one_sent.unsqueeze(0)), dim=0)
            # all_embeds = torch.mean(text_embeds, dim=1)

        return all_embeds
