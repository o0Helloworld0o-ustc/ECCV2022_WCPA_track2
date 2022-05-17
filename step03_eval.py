import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import pdb
import trimesh
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import torch

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from renderer import QuickRenderer

os.makedirs('output', exist_ok=True)
np.set_printoptions(suppress=True)






if __name__ == '__main__':

    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')


    opt = TestOptions().parse()  # get test options
    opt.isTrain = False
    opt.serial_batches = False
    opt.use_test_set = True
    opt.batch_size = 25



    test_dataset = create_dataset(opt)
    test_dataset_size = len(test_dataset)
    print('\nThe number of test images = %d' % test_dataset_size)


    model = create_model(opt)



    label_6dof_std = test_dataset.dataset.label_6dof_std
    label_6dof_mean = test_dataset.dataset.label_6dof_mean



    txt_path = Path(opt.data_root) / 'resources/projection_matrix.txt'
    M_proj = np.loadtxt(txt_path, dtype=np.float32)



    obj_path = Path(opt.data_root) / 'resources/example.obj'
    mesh = trimesh.load(obj_path, process=False)
    tris = np.array(mesh.faces, dtype=np.int32)


    img_h, img_w = 800, 800
    renderer = QuickRenderer(img_w, img_h, M_proj, tris)


    for i, data in enumerate(test_dataset):
        print('i =', i)
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()


        model.set_input(data)
        with torch.no_grad():
            model.forward()


        img_raw_list = data['img_raw'].numpy()
        pred_verts = model.pred_verts.view(-1, 1220, 3).cpu().numpy()
        pred_verts = pred_verts / 10.0      # denorm
        pred_6dof = model.pred_6dof.cpu().numpy()
        pred_6dof = pred_6dof * label_6dof_std + label_6dof_mean  # denorm


        fig, axes = plt.subplots(5, 5, figsize=(32, 32))

        for index, ax in enumerate(axes.flat):
            img_raw = img_raw_list[index]
            verts = pred_verts[index]

            R_t = np.identity(4)
            pitch_pred, yaw_pred, roll_pred = pred_6dof[index, :3]
            R_t[:3, :3] = Rotation.from_euler('yxz', [yaw_pred, pitch_pred, roll_pred], degrees=False).as_matrix().T
            R_t[3, :3] = pred_6dof[index, 3:]


            img_render = renderer(verts, R_t, overlap=img_raw)

            ax.imshow(img_render)
            ax.axis('off')

        output_path = 'output/%d.jpg' % i
        plt.savefig(output_path)
        plt.close(fig)


        if i == 5:
            break



    print('\n【Finish】')



