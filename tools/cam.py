import argparse
import os
import cv2
import numpy as np
import torch
#from torchvision import models

from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)

from pytorch_grad_cam import GuidedBackpropReLUModel

from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import _init_paths
from config import cfg
from config import update_config

import models

#import sys
#sys.path.insert(0, "/home/gruppo9/OnlineKD-HRNet-Human-Pose-Estimation-GradCAM/lib")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                         help='experiment configure file name',
                         required=True,
                         type=str)
    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    
    parser.add_argument(
        '--image-path',
        type=str,
        default='/home/gruppo9/OnlineKD-HRNet-Human-Pose-Estimation-GradCAM/data/babypose/images/test/000000004282.png',
        help='Input image path')
    
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    
    parser.add_argument('--method', type=str, default='eigencam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')

    parser.add_argument('--modelDir', type=str, default='', 
                        help='Directory of the pre-trained models')
    
    parser.add_argument('--logDir', type=str, default='',  # <-- Aggiungi anche questo
                        help='Directory to save the logs')
    
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

    update_config(cfg, args)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False).to(torch.device(args.device)).eval()

    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        best_model = "model_best.pth"
        model_state_file = os.path.join(cfg.OUTPUT_DIR, best_model)
        model.load_state_dict(torch.load(model_state_file), strict=False)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda().module

    #model = models.resnet50(pretrained=True).to(torch.device(args.device)).eval()

    # model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=False
    # )

    # if cfg.TEST.MODEL_FILE:
    #     logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    #     model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    # else:
    #     # model_state_file = os.path.join(
    #     #     final_output_dir, 'final_state.pth'
    #     # )
    #     best_model = -1
    #     for file in os.listdir(final_output_dir):
    #         if file == "model_best.pth":
    #             best_model = file
    #         # if "model_best" in file:
    #         #     if best_model == -1:
    #         #         best_model = file
    #         #     else:
    #         #         IndexError(f"Too many 'model_best' in dir {final_output_dir}")
    #     model_state_file = os.path.join(
    #         final_output_dir, best_model
    #         )
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda().module

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    target_layers = [model.stage4[-1].branches[-1][-1].conv2]
    # target_layers = [model.layer4]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(4)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        print(input_tensor.shape)
        print(targets)
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)
        
        

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    #gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
    #gb = gb_model(input_tensor, target_category=None)

    #cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    #cam_gb = deprocess_image(cam_mask * gb)
    #gb = deprocess_image(gb)

    os.makedirs(args.output_dir, exist_ok=True)

    cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
    #gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
    #cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

    cv2.imwrite(cam_output_path, cam_image)
    #cv2.imwrite(gb_output_path, gb)
    #cv2.imwrite(cam_gb_output_path, cam_gb)
