import os
from opt import get_opts


from models.nerf import Embedding, NeRF
from models.render import render_rays
from datasets import dataset_dict
from collections import defaultdict
from utils import *
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from metrics import *
from loss import loss_dict
from models.NeRF_system import NeRFSystem
import argparse
import yaml
from configs.config import parse_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file.", required=False, default='./configs/lego.yaml')
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")
    hparams = parse_args(parser)
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath='output/model',
                                          save_last=True,
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=2,
                                          )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [checkpoint_callback, pbar]
    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams['exp_name'],
        default_hp_metric=False
    )

    trainer = Trainer(max_epochs=hparams['num_epochs'],
                      callbacks=callbacks,
                      resume_from_checkpoint=hparams['ckpt_path'],
                      logger=logger,
                      gpus=hparams['num_gpus'],
                      strategy=DDPStrategy(find_unused_parameters=False) if hparams['num_gpus'] > 1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      # profiler
                      )

    trainer.fit(system)

