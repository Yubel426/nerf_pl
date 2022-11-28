import os
from abc import ABC
import torch
import numpy as np
import json
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from get_rays import get_rays


"""
input: transforms:
    {
        
        camera_angle_x  # focal = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        frames:
            {
                file_path
                rotation
                transform 4 * 4
            }    
    }
output: imgs, poses, render_poses, [H,W,f], split
"""
base_dir = 'D:/NeRF/datasets/data/nerf_synthetic/lego/'


class BlenderDataLoader(Dataset, ABC):

    def __init__(self, root_dir, split, img_wh):
        self.all_rays = None
        self.all_rgbs = None    # n*w*h,3
        self.image_paths = None
        self.poses = []     # n,3,4
        self.meta = None
        self.focal = None
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)
        w, h = self.img_wh
        self.focal = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])

        if self.split == 'test':  # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            # to test
            for frame in self.meta['frames'][0:2]:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                # Image.LANCZOS resize algorithm
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)  # RGBA
                img = transforms.ToTensor()(img)  # 4,h,w
                img = img.view(4, -1).permute(1, 0)  # 4,h*w -> h*w,4
                # r' = r*a + (1-a)
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # RGBA-> RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)] # (h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_rays = torch.cat(self.all_rays, 0)

    def __len__(self):

        return len(self.all_rays)

    def __getitem__(self, item):
        sample = {'rays': self.all_rays[item],
                  'rgbs': self.all_rgbs[item]}
        return sample


