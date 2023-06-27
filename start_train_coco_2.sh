python tools/train.py --cfg experiments/babypose/hrnet/4_stage_coco_dist_all4_lr_2.5e-4_kld_mse.yaml &&
mv emissions/emissions.csv emissions/4_stage_coco_dist_all4_lr_2.5e-4_kld_mse_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_coco_dist_all4_lr_2.5e-4_kld.yaml &&
mv emissions/emissions.csv emissions/4_stage_coco_dist_all4_lr_2.5e-4_kld_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_coco_dist_between_lr_2.5e-4_kld_mse.yaml &&
mv emissions/emissions.csv emissions/4_stage_coco_dist_between_lr_2.5e-4_kld_mse_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_coco_dist_between_lr_2.5e-4_kld.yaml &&
mv emissions/emissions.csv emissions/4_stage_coco_dist_between_lr_2.5e-4_kld_emissions.csv &&
echo "all trainings done" > done