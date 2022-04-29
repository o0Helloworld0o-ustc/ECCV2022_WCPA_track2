# ECCV2022 WCPA Challenge: From Face, Body and Fashion to 3D Virtual Avatars (Track2)

This repository contains tutorial codes for ["ECCV2022 WCPA Challenge Track2: Perspective Projection Based Monocular 3D Face Reconstruction"](https://tianchi.aliyun.com/competition/entrance/531961/information?spm=5176.12281976.0.0.3136f9319Ifv7e).



## 1. Data preparation


Follow the instructions of the competition website.

Download the data files and unpack them in the same folder as shown below.


```
wcpa_challenge_face_data
   ├─ image (contains 447,185 800x800 .jpg images of both training and test set)
   ├─ 68landmarks (contains 44,7185 .txt files of both training and test set)
   ├─ info (contains 356,640 .npz files of training set)
   ├─ list
   │    ├─ WCPA_track2_train_subject_id_list.txt
   │    ├─ WCPA_track2_test_subject_id_list.txt
   │    ├─ WCPA_track2_train.csv
   │    └─ WCPA_track2_test.csv
   └─ resources      
        ├─ example.obj
        ├─ kpt_ind.npy
        ├─ projection_matrix.txt
        └─ example_submission.npy
```





## 2. Understand the data format

Set up python environment.
```bash
conda create -n wcpa_track2 python=3.7
conda activate wcpa_track2
pip install -r requirements.txt
```

The visualization code depends on Sim3DR. Clone the code and compile it. (See https://github.com/cleardusk/3DDFA_V2/tree/master/Sim3DR)

Go to `visualization.ipynb` and learn about the detail of the data format.





## 3. Baseline

One simple baseline is to directly regress the 3D vertices and 6DoF (euler angles and translation vector) from the 800x800 image, leading to a score of 200~300 on the leaderboard. There is still much room for improvement.





