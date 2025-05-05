import time
from lib.core.evaluate import ConfusionMatrix,SegmentationMetric
from lib.core.general import non_max_suppression,check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,box_iou,coco80_to_coco91_class,plot_images,ap_per_class,output_to_target
from lib.utils.utils import time_synchronized
from lib.utils import plot_one_box,show_seg_result
import torch
import numpy as np
import pandas as pd

from pathlib import Path
import json
import random
import cv2
import os
import math
from torch.cuda import amp
from tqdm import tqdm

from lib.utils.utils import create_logger

from lib.core.Attacks.FGSM import fgsm_attack, fgsm_attack_with_noise, iterative_fgsm_attack
from lib.core.Attacks.JSMA import calculate_saliency, find_and_perturb_highest_scoring_pixels
from lib.core.Attacks.UAP import uap_sgd_yolop
from lib.core.Attacks.CCP import color_channel_perturbation

import sys
sys.path.append('/media/yunge/HDD1/REU-2024-YOLOP/pytorch-CycleGAN-and-pix2pix')  # Ensure the path is correct
from options.test_options import TestOptions
from models import create_model as create_pix2pix_model
from data.base_dataset import get_transform

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

def train(cfg, train_loader, model, criterion, optimizer, scaler, epoch, num_batch, num_warmup,
          writer_dict, logger, device, rank=-1):
    """
    train for one epoch

    Inputs:
    - config: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return total_loss, head_losses
    - writer_dict:
    outputs(2,)
    output[0] len:3, [1,3,32,32,85], [1,3,16,16,85], [1,3,8,8,85]
    output[1] len:1, [2,256,256]
    output[2] len:1, [2,256,256]
    target(2,)
    target[0] [1,n,5]
    target[1] [2,256,256]
    target[2] [2,256,256]
    Returns:
    None

    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()
    for i, (input, target, paths, shapes) in enumerate(train_loader):
        intermediate = time.time()
        num_iter = i + num_batch * (epoch - 1)

        if num_iter < num_warmup:
            # warm up
            lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                           (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
            xi = [0, num_warmup]
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_BIASE_LR if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(num_iter, xi, [cfg.TRAIN.WARMUP_MOMENTUM, cfg.TRAIN.MOMENTUM])

        data_time.update(time.time() - start)
        if not cfg.DEBUG:
            input = input.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            
        with amp.autocast(enabled=device.type != 'cpu'):
            outputs = model(input)
            total_loss, head_losses = criterion(outputs, target, shapes,model)

        # compute gradient and do update step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if rank in [-1, 0]:
            # measure accuracy and record loss
            losses.update(total_loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start)
            end = time.time()
            if i % cfg.PRINT_FREQ == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          speed=input.size(0)/batch_time.val,
                          data_time=data_time, loss=losses)
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

def validate(epoch, config, val_loader, val_dataset, model, criterion, output_dir, tb_log_dir, 
             perturbed_images=None, experiment_number=0, writer_dict=None, logger=None, device='cpu', 
             rank=-1, epsilon=None, attack_type=None, channel=None, step_decay = None, num_pixels = None, 
             uap=None, saliency_maps=None, perturb_type=None, use_pix2pix=False, pix2pix_params=None,
             resizer=None, quality=None, bit_depth=None, gauss=None, border_type=None, noise=None):
    
    results = []

    if attack_type is not None:
        # Constructing the save directory path with additional details based on the attack type
        if attack_type == 'FGSM':
            save_dir = os.path.join(output_dir, f'visualization_exp_{experiment_number}_epsilon_{epsilon}')
            perturbed_save_dir = os.path.join(output_dir, f'{attack_type}_perturbed_image_eps_{epsilon}_{time.strftime("%Y%m%d-%H%M%S")}_ExpNum{experiment_number}')

        elif attack_type == 'JSMA':
            save_dir = os.path.join(output_dir, f'visualization_exp_{experiment_number}_epsilon_{epsilon}_channel_{channel}')
            perturbed_save_dir = os.path.join(output_dir, f'{attack_type}_perturbed_image_eps_{epsilon}_channel_{channel}_{time.strftime("%Y%m%d-%H%M%S")}_ExpNum{experiment_number}')

        elif attack_type == 'UAP':
            save_dir = os.path.join(output_dir, f'visualization_exp_{experiment_number}_step_decay_{step_decay}')
            perturbed_save_dir = os.path.join(output_dir, f'{attack_type}_perturbed_image_step_decay_{step_decay}_{time.strftime("%Y%m%d-%H%M%S")}_ExpNum{experiment_number}')

        elif attack_type == 'CCP':
            save_dir = os.path.join(output_dir, f'visualization_exp_{experiment_number}_epsilon_{epsilon}_num_pixels_{num_pixels}_channel_{channel}')
            perturbed_save_dir = os.path.join(output_dir, f'{attack_type}_perturbed_image_eps_{epsilon}_num_pixels_{num_pixels}_channel_{channel}_{time.strftime("%Y%m%d-%H%M%S")}_ExpNum{experiment_number}')
        else:
            save_dir = output_dir

    else:
        save_dir = os.path.join(output_dir, f'visualization_NoAttack')
        perturbed_save_dir = os.path.join(output_dir, f'perturbed_image_{time.strftime("%Y%m%d-%H%M%S")}_ExpNum{experiment_number}')
    
    # Create the directories if they do not exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(perturbed_save_dir, exist_ok=True)
    
    # Create separate subdirectories for original and perturbed images
    original_images_dir = os.path.join(perturbed_save_dir, 'original_images')
    perturbed_images_dir = os.path.join(perturbed_save_dir, 'perturbed_images')
    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(perturbed_images_dir, exist_ok=True)
    
    # Create directory for metadata
    metadata_dir = os.path.join(perturbed_save_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    # Save pix2pix generated images
    pix2pix_dir = os.path.join(perturbed_save_dir, 'pix2pix_images')
    os.makedirs(pix2pix_dir, exist_ok=True)
            
    max_stride = 32
    weights = None
        
    _, imgsz = [check_img_size(x, s=max_stride) for x in config.MODEL.IMAGE_SIZE]
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(config.GPUS)
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(config.GPUS)
    training = False
    is_coco = False
    save_conf = False
    verbose = False
    save_hybrid = False
    log_imgs, wandb = min(16, 100), None
    nc = 1
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    try:
        import wandb
    except ImportError:
        wandb = None
        log_imgs = 0

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=model.nc)
    da_metric = SegmentationMetric(config.num_seg_class)
    ll_metric = SegmentationMetric(2)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    losses = AverageMeter()
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    T_inf = AverageMeter()
    T_nms = AverageMeter()

    model.eval()
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    pix2pix_model = None
    if use_pix2pix and pix2pix_params:
        # Save original command line arguments
        original_argv = sys.argv.copy()
        
        try:
            # Build command line arguments, using the parameters passed in
            cmd_args = ['test.py']
            
            # Must provide a dataroot, but it will not be used
            cmd_args.extend(['--dataroot', '/tmp'])
            
            # Add parameters from pix2pix_params
            for key, value in pix2pix_params.items():
                if value is not None:
                    if isinstance(value, bool) and value:
                        cmd_args.append(f'--{key}')
                    else:
                        cmd_args.extend([f'--{key}', str(value)])
            
            # Replace command line arguments
            sys.argv = cmd_args
            
            # Parse parameters and create model
            opt = TestOptions().parse()
            pix2pix_model = create_pix2pix_model(opt)
            pix2pix_model.setup(opt)
            pix2pix_model.eval()
            
            model_path = os.path.join(pix2pix_params['checkpoints_dir'], 
                                     pix2pix_params['name'], 
                                     f"{pix2pix_params['epoch']}_net_G.pth")
            print(f"Checking if model exists at: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            print(f"pix2pix_model={'exists' if pix2pix_model is not None else 'None'}")
            
        finally:
            sys.argv = original_argv

    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(val_loader), total=len(val_loader)):
        orig_shape, ((ratio_h, ratio_w), (pad_w, pad_h)) = shapes[0]
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = [tgt.to(device) for tgt in target]
            target = assign_target
            nb, _, height, width = img.shape
        if attack_type != None:
            for j in range(img.size(0)):
                orig_img_tensor = img[j].cpu().detach().numpy()
                img_filename = os.path.splitext(os.path.basename(paths[j]))[0]
                orig_img_path = os.path.join(original_images_dir, f'{img_filename}.npy')
                # Save as pt/npy format
                np.save(orig_img_path, orig_img_tensor)
            img.requires_grad = True
            det_out, da_seg_out, ll_seg_out = model(img)
            inf_out, train_out = det_out
            total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
            losses.update(total_loss.item(), img.size(0))
            total_loss.backward()
            data_grad = img.grad
            
            # Save metadata
            metadata = {
                'attack_type': attack_type,
                'epsilon': epsilon
            }

            if attack_type == 'FGSM':
                perturbed_data = fgsm_attack(img, epsilon, data_grad)
            elif attack_type == 'FGSM_WITH_NOISE':
                perturbed_data = fgsm_attack_with_noise(img, epsilon, data_grad)
            elif attack_type == 'ITERATIVE_FGSM':
                perturbed_data = iterative_fgsm_attack(img, epsilon, data_grad, alpha=0.01, num_iter=10, model=model, criterion=criterion, target=target, shapes=shapes)
            elif attack_type == 'CCP':
                perturbed_data = color_channel_perturbation(img, epsilon, data_grad, channel)
                metadata = {
                'attack_type': attack_type,
                'epsilon': epsilon,
                'channel': channel
            }

            elif attack_type == 'UAP':
                if epsilon == 0:
                    perturbed_data = img
                else:
                    # Apply UAP directly to the current batch
                    perturbed_data = img + uap
                metadata.update({'step_decay': step_decay})

            elif attack_type == 'JSMA':
                # Get the saliency map for the current batch
                batch_saliency = saliency_maps[batch_i * img.size(0):(batch_i + 1) * img.size(0)]
                # Apply JSMA to the current batch
                img_np = img.cpu().detach().numpy()
                perturbed_data_np, _ = find_and_perturb_highest_scoring_pixels(
                    img_np, batch_saliency, num_pixels, epsilon, 
                    perturbation_type=perturb_type
                )
                perturbed_data = torch.tensor(perturbed_data_np, dtype=torch.float32, device=device)
                
                metadata.update({
                    'num_pixels': num_pixels,
                    'perturb_type': perturb_type
                })
            else:
                perturbed_data = img

            img = perturbed_data

            def adjust_imagenet_norm_to_tanh_range(img):
                """Adjust ImageNet normalized data to the [-1,1] range"""
                if not isinstance(img, torch.Tensor):
                    return img
                
                # ImageNet mean and standard deviation
                means = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(img.device)
                stds = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(img.device)
                
                # Inverse normalization and map to [-1,1]
                img_unnormalized = img * stds + means  # Back to [0,1] range
                img_adjusted = img_unnormalized * 2 - 1  # Adjust to [-1,1] range
                
                return img_adjusted
            def denormalize_image(img):
                """Convert [-1,1] range tensor back to original range"""
                if not isinstance(img, torch.Tensor):
                    return img
                
                # First convert from [-1,1] to [0,1]
                img_unnormalized = (img + 1) / 2
                
                # ImageNet mean and standard deviation
                means = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(img.device)
                stds = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(img.device)
                
                # Apply inverse normalization
                img_normalized = (img_unnormalized - means) / stds
                
                return img_normalized
            
            # Save perturbed images
            for j in range(img.size(0)):
                # Save original numpy format
                img_np = img[j].cpu().detach().numpy()             
                img_filename = os.path.splitext(os.path.basename(paths[j]))[0]
                # img_path = os.path.join(perturbed_images_dir, f'{img_filename}.npy')
                # np.save(img_path, img_np)
                
                # Inverse normalization back to original range and save as jpg
                # ImageNet standardization parameters
                mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
                std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
                
                # Inverse normalization: (x * std) + mean
                img_unnormalized = (img_np * std) + mean
                
                # Clip to [0,1] range
                img_unnormalized = np.clip(img_unnormalized, 0, 1)
                
                # Convert to uint8 format [0,255]
                img_uint8 = (img_unnormalized.transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Convert to BGR (OpenCV format)
                img_uint8_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                
                # Save as jpg
                jpg_path = os.path.join(perturbed_images_dir, f'{img_filename}_attack.jpg')
                cv2.imwrite(jpg_path, img_uint8_bgr)
                
                metadata_path = os.path.join(metadata_dir, f'{img_filename}_metadata.json')

                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

            # Use pix2pix to process the attacked image
            if use_pix2pix and pix2pix_model is not None:
                print(f"Using pix2pix model for attack type: {attack_type}")
                img_normalized = adjust_imagenet_norm_to_tanh_range(img)
                pix2pix_input = {
                    'A': img_normalized,
                    'A_paths': [paths[j]]
                }
                
                pix2pix_model.set_input(pix2pix_input)
                # pix2pix_model.test()
                img = pix2pix_model.test().to(device)
                
                img = denormalize_image(img)

            if resizer:
                parts = resizer.split('x')
                width = int(parts[0])
                height = int(parts[1])
                defense_metadata = {'defense': 'resizing', 'size': f"{width}x{height}"}
                metadata.update(defense_metadata)
                
                # Process each image individually
                resized_data = []
                for i in range(perturbed_data.size(0)):
                    img_np = perturbed_data[i].detach().cpu().numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
                    # Downsample first and then upsample to eliminate attack traces
                    img_small = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)
                    img_back = cv2.resize(img_small, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                    img_tensor = torch.from_numpy(img_back.transpose(2, 0, 1)).float().to(device)  # [H,W,C] -> [C,H,W]
                    resized_data.append(img_tensor)
                perturbed_data = torch.stack(resized_data)

            # 2. (JPEG Compression)
            if quality is not None:
                quality = int(quality)  # Ensure it's an integer
                print(f"DEBUG: Applying JPEG compression with quality={quality}")
                imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
                imagenet_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
                
                defense_metadata = {'defense': 'jpeg_compression', 'quality': quality}
                metadata.update(defense_metadata)
                compressed_data = []
                for i in range(perturbed_data.size(0)):
                    img_normalized = perturbed_data[i].detach().cpu().numpy()
                    img_unnormalized = (img_normalized * imagenet_std) + imagenet_mean
                    img_unnormalized = np.clip(img_unnormalized, 0, 1)

                    img_unnorm_hwc = img_unnormalized.transpose(1, 2, 0)
                    img_uint8 = (img_unnorm_hwc * 255).astype(np.uint8)
                    
                    # 4. Apply JPEG compression
                    is_success, buffer = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not is_success:
                        print("ERROR: JPEG encoding failed")
                        compressed_data.append(perturbed_data[i])  # Use original image
                        continue
                    decoded_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                    decoded_float = decoded_img.astype(np.float32) / 255.0
                    decoded_chw = decoded_float.transpose(2, 0, 1)
                    decoded_normalized = (decoded_chw - imagenet_mean) / imagenet_std
                    img_tensor = torch.from_numpy(decoded_normalized).float().to(device)
                    compressed_data.append(img_tensor)
                
                # Ensure compressed data is not empty
                if compressed_data:
                    perturbed_data = torch.stack(compressed_data)
                    print(f"DEBUG: Final compressed shape={perturbed_data.shape}")
                else:
                    print("ERROR: No compressed data generated")

            # (Bit-Depth Reduction)
            if bit_depth is not None:
                bit_depth = int(bit_depth)
                print(f"DEBUG: Applying bit depth reduction with bit_depth={bit_depth}")
                
                defense_metadata = {'defense': 'bit_depth_reduction', 'bit_depth': bit_depth}
                metadata.update(defense_metadata)
                
                # ImageNet standardization mean and standard deviation
                imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
                imagenet_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
                
                reduced_data = []
                for i in range(perturbed_data.size(0)):
                    img_normalized = perturbed_data[i].detach().cpu().numpy()
                    img_unnormalized = (img_normalized * imagenet_std) + imagenet_mean

                    img_unnormalized = np.clip(img_unnormalized, 0, 1)
                    factor = 2**(8 - bit_depth)
                    img_quantized = np.floor(img_unnormalized * 255 / factor) * factor / 255
                    img_renormalized = (img_quantized - imagenet_mean) / imagenet_std
                    img_tensor = torch.from_numpy(img_renormalized).float().to(device)
                    reduced_data.append(img_tensor)
                
                perturbed_data = torch.stack(reduced_data)
            # 4. (Gaussian Blurring)
            if gauss:
                parts = gauss.split('x')
                ksize_w = int(parts[0])
                ksize_h = int(parts[1])
                # Ensure the kernel size is odd
                ksize_w = ksize_w if ksize_w % 2 == 1 else ksize_w + 1
                ksize_h = ksize_h if ksize_h % 2 == 1 else ksize_h + 1
                
                # Get boundary type
                border_type_map = {
                    'default': cv2.BORDER_DEFAULT,
                    'constant': cv2.BORDER_CONSTANT,
                    'reflect': cv2.BORDER_REFLECT,
                    'replicate': cv2.BORDER_REPLICATE
                }
                border_type = border_type_map.get(border_type, cv2.BORDER_DEFAULT)
                
                defense_metadata = {'defense': 'gaussian_blur', 'kernel_size': f"{ksize_w}x{ksize_h}", 'border_type': border_type}
                metadata.update(defense_metadata)
                
                blurred_data = []
                for i in range(perturbed_data.size(0)):
                    img_np = perturbed_data[i].detach().cpu().numpy().transpose(1, 2, 0)
                    # Apply Gaussian blur to image
                    blurred_img = cv2.GaussianBlur(img_np, (ksize_w, ksize_h), 0, borderType=border_type)
                    img_tensor = torch.from_numpy(blurred_img.transpose(2, 0, 1)).float().to(device)
                    blurred_data.append(img_tensor)
                perturbed_data = torch.stack(blurred_data)

            # 5. (Noise Generation / Noise Addition)
            if noise is not None:
                sigma = abs(noise)  # Noise standard deviation
                defense_metadata = {'defense': 'noise_addition', 'sigma': sigma}
                metadata.update(defense_metadata)
                
                noise_data = []
                for i in range(perturbed_data.size(0)):
                    img_np = perturbed_data[i].detach().cpu().numpy()
                    # Generate Gaussian noise and add to image
                    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
                    noisy_img = img_np + noise # Limiting the range to [0,1]
                    img_tensor = torch.from_numpy(noisy_img).float().to(device)
                    noise_data.append(img_tensor)
                perturbed_data = torch.stack(noise_data)

            # Finally assign the processed image to img
            img = perturbed_data
        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w = int(pad_w)
            pad_h = int(pad_h)
            ratio = shapes[0][1][0][0]
            t = time_synchronized()
            det_out, da_seg_out, ll_seg_out= model(img)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0),img.size(0))

            inf_out,train_out = det_out

            #driving area segment evaluation
            _,da_predict=torch.max(da_seg_out, 1)
            _,da_gt=torch.max(target[1], 1)
            da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            da_metric.reset()
            da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
            da_acc = da_metric.pixelAccuracy()
            da_IoU = da_metric.IntersectionOverUnion()
            da_mIoU = da_metric.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,img.size(0))
            da_IoU_seg.update(da_IoU,img.size(0))
            da_mIoU_seg.update(da_mIoU,img.size(0))

            #lane line segment evaluation
            _,ll_predict=torch.max(ll_seg_out, 1)
            _,ll_gt=torch.max(target[2], 1)
            ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
            ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

            ll_metric.reset()
            ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
            ll_acc = ll_metric.lineAccuracy()
            ll_IoU = ll_metric.IntersectionOverUnion()
            ll_mIoU = ll_metric.meanIntersectionOverUnion()

            ll_acc_seg.update(ll_acc,img.size(0))
            ll_IoU_seg.update(ll_IoU,img.size(0))
            ll_mIoU_seg.update(ll_mIoU,img.size(0))
            
            total_loss, head_losses = criterion((train_out,da_seg_out, ll_seg_out), target, shapes,model)   
            losses.update(total_loss.item(), img.size(0))

            #NMS
            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            output = non_max_suppression(inf_out, conf_thres= config.TEST.NMS_CONF_THRESHOLD, iou_thres=config.TEST.NMS_IOU_THRESHOLD, labels=lb)
            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))

            # Visualizations
            if config.TEST.PLOTS:
                if batch_i == 0:
                    for i in range(test_batch_size):
                        img_filename = os.path.splitext(os.path.basename(paths[i]))[0] + '.jpg'
                        img_path = os.path.join(config.DATASET.DATAROOT, img_filename)
                        
                        if not os.path.exists(img_path):
                            img_path = paths[i]
                        print(f"Loading image from: {img_path}")
                        img_test = cv2.imread(img_path)
                        # Check if image was loaded successfully
                        if img_test is None:
                            print(f"Error: Could not load image from {img_path}. Skipping visualization for this image.")
                            break

                        da_seg_mask = da_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)

                        da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_seg_mask = torch.max(da_seg_mask, 1)

                        da_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        da_gt_mask = torch.nn.functional.interpolate(da_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, da_gt_mask = torch.max(da_gt_mask, 1)

                        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                        da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()

                        img_test1 = img_test.copy()
                        
                        _ = show_seg_result(img_test, da_seg_mask, i, epoch,save_dir)
                        _ = show_seg_result(img_test1, da_gt_mask, i, epoch, save_dir, is_gt=True)

                        img_ll = cv2.imread(img_path)
                        ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_seg_mask = torch.max(ll_seg_mask, 1)

                        ll_gt_mask = target[2][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                        ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                        _, ll_gt_mask = torch.max(ll_gt_mask, 1)

                        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                        ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()

                        img_ll1 = img_ll.copy()
                        _ = show_seg_result(img_ll, ll_seg_mask, i,epoch,save_dir, is_ll=True)
                        _ = show_seg_result(img_ll1, ll_gt_mask, i, epoch, save_dir, is_ll=True, is_gt=True)

                        img_det = cv2.imread(img_path) 
                        
                        # Check if image was loaded successfully
                        if img_det is None:
                            print(f"Error: Could not load image from {img_path}. Skipping visualization for this image.")
                            continue
                            
                        img_gt = img_det.copy()
                        det = output[i].clone()
                        
                        if len(det):
                            det[:,:4] = scale_coords(img[i].shape[1:],det[:,:4],img_det.shape).round()
                        for *xyxy,conf,cls in reversed(det):
                            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img_det , label=label_det_pred, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir+"/batch_{}_{}_det_pred.png".format(epoch,i),img_det)

                        labels = target[0][target[0][:, 0] == i, 1:]
                        labels[:,1:5]=xywh2xyxy(labels[:,1:5])
                        if len(labels):
                            labels[:,1:5]=scale_coords(img[i].shape[1:],labels[:,1:5],img_gt.shape).round()
                        for cls,x1,y1,x2,y2 in labels:
                            label_det_gt = f'{names[int(cls)]}'
                            xyxy = (x1,y1,x2,y2)
                            plot_one_box(xyxy, img_gt , label=label_det_gt, color=colors[int(cls)], line_thickness=3)
                        cv2.imwrite(save_dir+"/batch_{}_{}_det_gt.png".format(epoch,i),img_gt)
        
        for si, pred in enumerate(output):
            labels = target[0][target[0][:, 0] == si, 1:]     # all object in one image 
            nl = len(labels)    # num of object
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if config.TEST.SAVE_TXT:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging
            if config.TEST.PLOTS and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            if config.TEST.SAVE_JSON:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})


            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if config.TEST.PLOTS:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):                    
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # n*m  n:pred  m:label
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if config.TEST.PLOTS and batch_i < 3:
            f = save_dir +'/'+ f'test_batch{batch_i}_labels.jpg'  # labels
            f = save_dir +'/'+ f'test_batch{batch_i}_pred.jpg'  # predictions

    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(),ap75.mean(),ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc <= 20 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    
    # Plots
    if config.TEST.PLOTS:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if config.TEST.SAVE_JSON and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in val_loader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if config.TEST.SAVE_TXT else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    
    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]
    
    metric_result = {
        'attack_type': attack_type if attack_type else 'Baseline',
        'epsilon': epsilon,
        'num_pixels': num_pixels,
        'channel': channel,
        'step_decay': step_decay,
        'total_loss': losses.avg,
        'da_seg_acc': da_acc_seg.avg,
        'da_seg_iou': da_IoU_seg.avg,
        'da_seg_miou': da_mIoU_seg.avg,
        'll_seg_acc': ll_acc_seg.avg,
        'll_seg_iou': ll_IoU_seg.avg,
        'll_seg_miou': ll_mIoU_seg.avg,
        'p': mp,
        'r': mr,
        'map50': map50,
        'map': map,
        't_inf': T_inf.avg,
        't_nms': T_nms.avg
    }
    
    results.append(metric_result)

    # Save results to CSV
    save_results_to_csv(results, f'validation_results_{time.strftime("%Y%m%d-%H%M%S")}.csv', save_dir)
    
    return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t

def save_results_to_csv(results, file_name, directory='.'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"Saved results to {file_path}")