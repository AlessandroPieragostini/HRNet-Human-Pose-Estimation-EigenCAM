
# Online Knowledge Distillation on HRNet with EigenCAM for explainable AI

## Introduction
This project is a *fork* of [https://github.com/leoxiaobin/deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch). We implemented a variant of *pose_hrnet* model called *multi_out_pose_hrnet* which produces an output heatmap for each stage of original HRNet. In addition, we introduced the possibility to activate Online Knowledge Distillation between different stages, inspired by [https://github.com/zhengli97/OKDHP](https://github.com/zhengli97/OKDHP). 

All our tests were carried out on [BabyPose dataset](https://link.springer.com/article/10.1007/s11517-022-02696-9).

## Notes
Python 3.9 environment and NVIDIA GPUs are needed. Other platforms or GPU cards are not fully tested.


## Quick start
### Installation
1. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
2. Install dependencies in your Python 3.9 environment:
   ```
   pip install -r requirements.txt
   ```
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools
   ├── visualization
   ├── README.md
   └── requirements.txt
   ```

5. Download pretrained models from model zoo([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ)). In particular, be sure to download 'pose_hrnet_w32_256x192.pth' and extract it in project folder like this:
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- pose_coco
                |-- pose_hrnet_w32_256x192.pth

   ```

All experiments use this model.

### Data preparation

**For BabyPose data**, please download from [BabyPose download](https://mega.nz/file/434XTAQC#hfFmccK7TkBeUcywf9fp6fSzrWbxexlzXJ3ngTRRU6U), extract files under {POSE_ROOT}/data and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- babypose
    `-- |-- annotations
        |   |-- person_keypoints_train.json
        |   |-- person_keypoints_val.json
        |   |-- person_keypoints_test.json
        |-- person_detection_results
        |   |-- result_train.json
        |   |-- result_train.json
        |   |-- result_train.json
        `-- images
            |-- train
            |-- val
            |-- test
```
### Trained models
Download [here](https://mega.nz/file/omAQBIaY#U0XgNaD4vw5NwX7m1_Dz-DlOrGcxUWA8O5goHe16K2s) our trained models for each experiment and extract them in {POSE_ROOT}/models/pytorch.

### Configuration
Compared to [main work](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), new configuration options are available:

- MODEL.N_STAGE (default: 4): number of stages of HRNet. If number lower than 4 is specified, HRNet is truncated;
- MODEL.MULTI (default: false): if true, output heatmaps of each stage are considered. Set true for applying knowledge distillation;
- LOSS.USE_KLD (default: false): set true to add soft loss term (Kullback Leibler) to total loss.
- LOSS.USE_MSE (default: false): set true to add hard loss term (MSE) to total loss.
- LOSS.KLD_COUPLES: array of couples of stages (indexes) of HRNet in the form "(student, teacher)". For each couple, knowledge will be distilled from teacher to student.
- TRAIN.KLD_WEIGHT (default : 1): weight referred to soft loss
- TRAIN.TEACHER_WEIGHT (default: 1): weight referred to teacher loss
### Training and Testing

#### Testing on BabyPose dataset

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/<experiment_name>.yaml \
    TEST.MODEL_FILE models/pytorch/<experiment_name>.pth
```

#### Training on BabyPose dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrnet/<experiment_name>.yaml
```

### Contributors
[Alessandro Pieragostini](https://github.com/AlessandroPieragostini) & [Kevin Javier](https://github.com/sup3rk24) 
