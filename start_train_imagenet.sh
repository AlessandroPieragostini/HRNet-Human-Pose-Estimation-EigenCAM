python tools/train.py --cfg experiments/babypose/hrnet/3_stage_imagenet.yml &&
mv emissions/emissions.csv emissions/3_stage_imagenet_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_imagenet.yml &&
mv emissions/emissions.csv emissions/4_stage_imagenet_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_dist_34_imagenet.yml &&
mv emissions/emissions.csv emissions/4_stage_dist_34_imagenet_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_dist_all4_imagenet.yml &&
mv emissions/emissions.csv emissions/4_stage_dist_all4_imagenet_emissions.csv &&
python tools/train.py --cfg experiments/babypose/hrnet/4_stage_dist_between_imagenet.yml &&
mv emissions/emissions.csv emissions/4_stage_dist_between_imagenet_emissions.csv &&
echo "all trainings done"