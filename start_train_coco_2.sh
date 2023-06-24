python tools/train.py --cfg experiments/babypose/hrnet/3_stage_coco_lr_1e-3.yaml &&
mv emissions/emissions.csv emissions/3_stage_coco_lr_1e-3.yaml_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_coco_dist_34_lr_1e-3_kld_mse.yaml &&
mv emissions/emissions.csv emissions/4_stage_coco_dist_34_lr_1e-3_kld_mse_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_coco_dist_34_lr_1e-3_kld.yaml &&
mv emissions/emissions.csv emissions/4_stage_coco_dist_34_lr_1e-3_kld_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_coco_lr_1e-3.yaml &&
mv emissions/emissions.csv emissions/4_stage_coco_lr_1e-3_emissions.csv &&
echo "all trainings done" > done