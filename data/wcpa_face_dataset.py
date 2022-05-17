import cv2
import pdb
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation

import torch
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset





class WCPAFaceDataset(BaseDataset):

    def __init__(self, opt):
        self.opt = opt
        self.data_root = Path(opt.data_root)
        self.is_train = opt.isTrain
        self.img_size = opt.img_size
        self.use_test_set = opt.use_test_set

        if self.use_test_set:
            csv_path = self.data_root / 'list/WCPA_track2_test.csv'
            self.df = pd.read_csv(csv_path, dtype={'subject_id': str, 'facial_action': str, 'img_id': str})

        else:
            if self.is_train:
                self.df = pd.read_csv(opt.csv_path_train, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                    nrows=3333 if opt.debug else None)
            else:
                self.df = pd.read_csv(opt.csv_path_val, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                    nrows=1111 if opt.debug else None)


        img_mean = np.array([0.5, 0.5, 0.5])
        img_std = np.array([0.5, 0.5, 0.5])
        self.tfm_std = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

        self.dst_pts = np.float32([
            [0, 0],
            [0, opt.img_size - 1],
            [opt.img_size - 1, 0]
        ])


        npz_path = opt.label_mean_std
        M = np.load(npz_path)
        self.label_6dof_mean = M['label_6dof_mean']
        self.label_6dof_std = M['label_6dof_std']



    def __getitem__(self, index):
        subject_id = self.df['subject_id'][index]
        facial_action = self.df['facial_action'][index]
        img_id = self.df['img_id'][index]

        img_path = self.data_root / 'image' / subject_id / facial_action / f'{img_id}_ar.jpg'
        npz_path = self.data_root / 'info' / subject_id / facial_action / f'{img_id}_info.npz'
        txt_path = self.data_root / '68landmarks' / subject_id / facial_action / f'{img_id}_68landmarks.txt'

        img_raw = cv2.imread(str(img_path))
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_raw.shape
        pts68 = np.loadtxt(txt_path, dtype=np.int32)


        x_min, y_min = pts68.min(axis=0)
        x_max, y_max = pts68.max(axis=0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        size = max(w, h)
        ss = np.array([0.75, 0.75, 0.85, 0.65])     # predefined expand size

        left = x_center - ss[0] * size
        right = x_center + ss[1] * size
        top = y_center - ss[2] * size
        bottom = y_center + ss[3] * size

        src_pts = np.float32([
            [left, top],
            [left, bottom],
            [right, top]
        ])

        tform = cv2.getAffineTransform(src_pts, self.dst_pts)
        img_local = cv2.warpAffine(img_raw, tform, (self.img_size,)*2)
        img_local_tensor = self.tfm_std(Image.fromarray(img_local))


        img_global = img_raw.copy()
        left, top, right, bottom = self.check_boundary(left, top, right, bottom, img_w, img_h)
        img_global[:, :left] = 0
        img_global[:, right:] = 0
        img_global[:top] = 0
        img_global[bottom:] = 0

        img_global = cv2.resize(img_global, (self.img_size,)*2)
        img_global_tensor = self.tfm_std(Image.fromarray(img_global))

        d = {
            'img_local': img_local_tensor,
            'img_global': img_global_tensor
        }


        if self.use_test_set:
            d['img_raw'] = img_raw

        else:
            M = np.load(npz_path)

            yaw_gt, pitch_gt, roll_gt = Rotation.from_matrix(M['R_t'][:3, :3].T).as_euler('yxz', degrees=False)
            label_euler = np.array([pitch_gt, yaw_gt, roll_gt])
            label_translation = M['R_t'][3, :3]

            label_6dof = np.concatenate([label_euler, label_translation])
            label_6dof = (label_6dof - self.label_6dof_mean) / self.label_6dof_std
            label_6dof_tensor = torch.tensor(label_6dof, dtype=torch.float32)

            label_verts = M['verts'] * 10.0     # roughly [-1, 1]
            label_verts_tensor = torch.tensor(label_verts, dtype=torch.float32)

            d['label_verts'] = label_verts_tensor
            d['label_6dof'] = label_6dof_tensor

        return d



    def __len__(self):
        return len(self.df)



    def check_boundary(self, left, top, right, bottom, img_w, img_h):
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        if left < 0:
            left = 0

        if top < 0:
            top = 0

        if right > img_w:
            right = img_w - 1

        if bottom > img_h:
            bottom = img_h - 1

        return left, top, right, bottom
    


