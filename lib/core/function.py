# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_pred_debug_images, save_gt_debug_images


logger = logging.getLogger(__name__)

def get_current_consistency_weight(current, rampup_length):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def train(config, train_loader, model, criterion, criterion_kld, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    roles = [ 'Teacher_' if i == config.MODEL.N_STAGE-1 else f'Student{i+1}' 
                for i in range(config.MODEL.N_STAGE) ]
    loss_by_stage = [ AverageMeter() for i in range(config.MODEL.N_STAGE) ]
    acc_by_stage = [ AverageMeter() for i in range(config.MODEL.N_STAGE) ]

    losses = AverageMeter()
    pred_to_plot = []


    teacher_weight = config.TRAIN.TEACHER_WEIGHT
    kld_weight = config.TRAIN.KLD_WEIGHT
    cons_weight = get_current_consistency_weight(epoch, config.TRAIN.LENGTH)

    #switch to train mode
    model.train()

    end = time.time()

    print("Training\n---------------------")

    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # compute outputs = outputs from stages
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        
        #distillating from teacher to all students
        loss_hard = 0
        loss_soft = 0
        teacher_loss = 0

        if isinstance(outputs, list):
            for index, output in enumerate(outputs):                
                if index == len(outputs) - 1:
                    teacher_loss = criterion(output, target, target_weight)
                    teacher_loss *= teacher_weight
                    stage_loss = teacher_loss
                else:
                    lh = criterion(output, target, target_weight)
                    
                    if config.LOSS.USE_KLD: 
                        ls = criterion_kld(output, outputs[-1], target_weight)
                        ls *= kld_weight * cons_weight
                    else:
                        ls = 0

                    loss_hard += lh
                    loss_soft += ls
                    stage_loss = lh + ls
                    
                loss_by_stage[index].update(stage_loss.item(), input.size(0))
                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
                pred_to_plot.append(pred)
                acc_by_stage[index].update(avg_acc, cnt)
        else:
            raise ValueError("Model output is not a list")

        loss = loss_hard + teacher_loss + loss_soft

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            #total loss and teacher accuracy
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            msg += "".join(["{role} Acc {acc.val:.5f} ({acc.avg:.5f})  ".format(
                    role = roles[i][0] + roles[i][-1], acc = acc_by_stage[i]
            ) for i in range(config.MODEL.N_STAGE) ])

            logger.info(msg)
            prefix = '{}_ep{}_b{}'.format(os.path.join(output_dir, 'val'), epoch, i)

            save_gt_debug_images(config, input, meta, target, prefix)
                
            for index in range(config.MODEL.N_STAGE):
                save_pred_debug_images(config, input, meta, outputs[index], pred_to_plot[index]*4, prefix  + f"_{roles[index]}")
        
    #once for epoch
    writer = writer_dict['writer']

    for index in range(config.MODEL.N_STAGE):
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar( roles[index] + '_train_loss', loss_by_stage[index].val, global_steps)
        writer.add_scalar( roles[index] + '_train_acc', acc_by_stage[index].val, global_steps)
    
    writer.add_scalar('train_loss', losses.val, global_steps)
    writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, val_dataset, model, criterion, criterion_kld,  output_dir,
             tb_log_dir, writer_dict=None, epoch = 0):
    
    batch_time = AverageMeter()
    
    roles = [ 'Teacher_' if i == config.MODEL.N_STAGE-1 else f'Student{i+1}' 
                for i in range(config.MODEL.N_STAGE) ]
    loss_by_stage = [ AverageMeter() for i in range(config.MODEL.N_STAGE) ]
    acc_by_stage = [ AverageMeter() for i in range(config.MODEL.N_STAGE) ]
    pred_to_plot = []
    
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    num_samples = len(val_dataset)

    # N_STAGE corrisponde al numero di stage e output
    all_preds = np.zeros(
        (config.MODEL.N_STAGE, num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    
    # N_STAGE corrisponde al numero di stage e output
    all_boxes = np.zeros((config.MODEL.N_STAGE, num_samples, 6))
    
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    teacher_weight = config.TRAIN.TEACHER_WEIGHT
    kld_weight = config.TRAIN.KLD_WEIGHT
    cons_weight = get_current_consistency_weight(epoch, config.TRAIN.LENGTH)

    print("Validation\n---------------------")
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute outputs = teacher_out, stud_out
            outputs = model(input)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            num_images = input.size(0)
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            

            #distillating from teacher to all students
            loss_hard = 0
            loss_soft = 0
            teacher_loss = 0

            if isinstance(outputs, list):
                for index, output in enumerate(outputs):                
                    if index == len(outputs) - 1:
                        teacher_loss = criterion(output, target, target_weight)
                        teacher_loss *= teacher_weight
                        stage_loss = teacher_loss
                    else:
                        lh = criterion(output, target, target_weight)
                        
                        if config.LOSS.USE_KLD: 
                            ls = criterion_kld(output, outputs[-1], target_weight)
                            ls *= kld_weight * cons_weight
                        else:
                            ls = 0

                        loss_hard += lh
                        loss_soft += ls
                        stage_loss = lh + ls
                        
                    loss_by_stage[index].update(stage_loss.item(), input.size(0))
                    _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
                    pred_to_plot.append(pred)
                    acc_by_stage[index].update(avg_acc, cnt)
                    preds, maxvals = get_final_preds(
                    config, output.clone().cpu().numpy(), c, s)

                    all_preds[index][idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                    all_preds[index][idx:idx + num_images, :, 2:3] = maxvals
                    # double check this all_boxes parts
                    all_boxes[index][idx:idx + num_images, 0:2] = c[:, 0:2]
                    all_boxes[index][idx:idx + num_images, 2:4] = s[:, 0:2]
                    all_boxes[index][idx:idx + num_images, 4] = np.prod(s*200, 1)
                    all_boxes[index][idx:idx + num_images, 5] = score
            else:
                raise ValueError("Model output is not a list")

            loss = loss_hard + teacher_loss + loss_soft
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            image_path.extend(meta['image'])
            idx += num_images

            if i % config.PRINT_FREQ == 0:
                #total loss and teacher accuracy
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                        epoch, i, len(val_loader), batch_time=batch_time, loss=losses)
                msg += "".join(["{role} Acc {acc.val:.5f} ({acc.avg:.5f})  ".format(
                        role = roles[i][0] + roles[i][-1], acc = acc_by_stage[i]
                ) for i in range(config.MODEL.N_STAGE) ])
                logger.info(msg)

                prefix = '{}_ep{}_b{}'.format(os.path.join(output_dir, 'val'), epoch, i)

                save_gt_debug_images(config, input, meta, target, prefix)
                
                for index in range(config.MODEL.N_STAGE):
                    save_pred_debug_images(config, input, meta, outputs[index], pred_to_plot[index]*4, prefix  + f"_{roles[index]}")
                    
        #once for epoch
        model_name = config.MODEL.NAME
        perf_indicators = [0]*config.MODEL.N_STAGE
        prefix = '{}_ep{}'.format(os.path.join(output_dir, 'val'), epoch)
        
        writer = writer_dict['writer']

        for index in range(config.MODEL.N_STAGE):
            
            name_values, perf_indicators[index] = val_dataset.evaluate(
                config, all_preds[index], output_dir, all_boxes[index], image_path,
                filenames, imgnums
            )

            print(roles[index], "\n")

            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, model_name)
            else:
                _print_name_value(name_values, model_name)

            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar( roles[index] + '_valid_loss', loss_by_stage[index].avg, global_steps)
            writer.add_scalar( roles[index] + '_valid_acc', acc_by_stage[index].avg, global_steps)

            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars( roles[index] + '_valid', dict(name_value), global_steps)
            else:
                writer.add_scalars( roles[index] + '_valid', dict(name_values), global_steps)

        writer.add_scalar('valid_loss', losses.val, global_steps)
        writer_dict['valid_global_steps'] += 1

        return perf_indicators


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
