
from models.nerf import Embedding, NeRF
from models.render import render_rays
from datasets import dataset_dict
from collections import defaultdict
from utils.visualization import *
from utils import *
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from metrics import *
from loss import MSELoss


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()

        self.save_hyperparameters(hparams)

        # self.loss = loss_dict[hparams.loss_tpye]
        self.loss = MSELoss()
        self.embedding_xyz = Embedding(3, 10)

        self.embedding_dir = Embedding(3, 4)
        self.embeddings = [self.embedding_xyz, self.embedding_dir]
        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams['N_importance'] > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays']
        rgbs = batch['rgbs']
        return rays, rgbs
# TODO: 写进 init 中
    def forward(self, rays):
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams['chunk']):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams['chunk']],
                            self.hparams['N_samples'],
                            self.hparams['use_disp'],
                            self.hparams['perturb'],
                            self.hparams['noise_std'],
                            self.hparams['N_importance'],
                            self.hparams['chunk'],  # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):

        dataset = dataset_dict[self.hparams['dataset_name']]
        kwargs = {'root_dir': self.hparams['root_dir'],
                  'img_wh': tuple(self.hparams['img_wh'])}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams['spheric_poses']
            kwargs['val_num'] = self.hparams['num_gpus']
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams['batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        self.log('train/psnr', psnr_)
        self.log('train/loss', loss)
        self.log('lr', get_learning_rate(self.optimizer))
        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)

        log = {'val/loss': self.loss(results, rgbs), 'val/psnr': psnr(results[f'rgb_{typ}'], rgbs)}
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)




