import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.MultiPR2Gen import R2GenGPT
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy


def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)
    
    if 'ddp' == args.strategy:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = args.strategy
    debug = ''   # 'debug' or ''
    # Build trainer
    if debug == 'debug':
        trainer = pl.Trainer(fast_dev_run=2) # True (1) or number of batches
    else:
        trainer = pl.Trainer(
            devices=args.devices,
            num_nodes=args.num_nodes,
            strategy=strategy,
            accelerator=args.accelerator,
            precision=args.precision,
            val_check_interval = args.val_check_interval,
            limit_val_batches = args.limit_val_batches,
            limit_train_batches = args.limit_train_batches,
            limit_test_batches = args.limit_test_batches,
            max_epochs = args.max_epochs,
            num_sanity_val_steps = args.num_sanity_val_steps,
            accumulate_grad_batches=args.accumulate_grad_batches,
            callbacks=callbacks["callbacks"], 
            logger=callbacks["loggers"]
        )

    if args.ckpt_file is not None:
        model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = R2GenGPT(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)
    print('end')


if __name__ == '__main__':
    main()