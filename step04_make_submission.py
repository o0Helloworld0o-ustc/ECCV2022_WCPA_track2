import os
import sys
import pdb
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import torch

from data import create_dataset
from models import create_model
from options.test_options import TestOptions

os.makedirs('output', exist_ok=True)
np.set_printoptions(suppress=True)




def batch_euler2matrix(batch_euler):
    n = batch_euler.shape[0]
    assert batch_euler.shape[1] == 3
    batch_matrix = np.zeros([n, 3, 3], dtype=np.float32)

    for i in range(n):
        pitch, yaw, roll = batch_euler[i]
        R = Rotation.from_euler('yxz', [yaw, pitch, roll], degrees=False).as_matrix().T
        batch_matrix[i] = R

    return batch_matrix






if __name__ == '__main__':

    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')


    opt = TestOptions().parse()  # get test options
    opt.isTrain = False
    opt.serial_batches = True
    opt.use_test_set = True



    test_dataset = create_dataset(opt)
    test_dataset_size = len(test_dataset)
    print('\nThe number of test images = %d' % test_dataset_size)


    model = create_model(opt)



    label_6dof_std = test_dataset.dataset.label_6dof_std
    label_6dof_mean = test_dataset.dataset.label_6dof_mean




    prediction = np.zeros([test_dataset_size, 1220 * 3 + 9 + 3], dtype=np.float32)


    index = 0
    assert opt.serial_batches == True

    for i, data in enumerate(test_dataset):
        if i % 20 == 0:
            print('i =', i)

        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()


        model.set_input(data)
        with torch.no_grad():
            model.forward()


        pred_verts = model.pred_verts.view(-1, 1220, 3).cpu().numpy()
        pred_verts = pred_verts / 10.0      # denorm
        pred_6dof = model.pred_6dof.cpu().numpy()
        pred_6dof = pred_6dof * label_6dof_std + label_6dof_mean  # denorm


        cur_batch_size = pred_verts.shape[0]
        prediction[index:index+cur_batch_size, :3660] = pred_verts.reshape(cur_batch_size, 1220 * 3)
        prediction[index:index+cur_batch_size, 3660:3660+9] = batch_euler2matrix(pred_6dof[:, :3]).reshape(cur_batch_size, 9)
        prediction[index:index+cur_batch_size, -3:] = pred_6dof[:, -3:]
        index += cur_batch_size


        if i == -1:
            break


    output_path = 'output/my_submission.npy'
    np.save(output_path, prediction)


    print('\n【Finish】')




