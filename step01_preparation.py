import os
import pdb
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
os.makedirs('cache', exist_ok=True)
seed = 0






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    opt = parser.parse_args()


    data_root = Path(opt.data_root)
    csv_path = data_root / 'list/WCPA_track2_train.csv'
    df = pd.read_csv(csv_path, dtype={'subject_id': str, 'facial_action': str, 'img_id': str})


    # We use 80% subjects as training set and 20% subject% as validation set.
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)

    subject_list = df['subject_id'].unique()
    train_index, val_index = next(kf.split(subject_list))
    print('\nlen(train_index) =', len(train_index))
    print('len(val_index) =', len(val_index))
    print('\ntrain_index =', train_index)
    print('val_index =', val_index)


    train_subjects = subject_list[train_index]
    val_subjects = subject_list[val_index]

    df['is_train'] = df['subject_id'].isin(train_subjects)

    c = df['is_train'] == True
    df[c].to_csv('cache/train_list.csv', index=False)
    df[~c].to_csv('cache/val_list.csv', index=False)

    print('\nnumber of instances in train_set:', c.sum())
    print('number of instances in val_set:', (~c).sum())



    # We precomupte the statics of the label
    label_6dof_mean = [-0.018197, -0.017891, 0.025348, -0.005368, 0.001176, -0.532206]   # mean of pitch, yaw, roll, tx, ty, tz
    label_6dof_std = [0.314015, 0.271809, 0.081881, 0.022173, 0.048839, 0.065444]        # std of pitch, yaw, roll, tx, ty, tz
    d = {'label_6dof_mean': label_6dof_mean, 'label_6dof_std': label_6dof_std}
    np.savez('cache/label_mean_std.npz', **d)


    print('\n【Finish】')



