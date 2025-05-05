import argparse
import os, sys
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import torch
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from lib.utils import DataLoaderX

from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import validate

from lib.models import get_net
from lib.utils.utils import create_logger, select_device, create_experiment_logger

from lib.core.Attacks.JSMA import calculate_saliency, find_and_perturb_highest_scoring_pixels
from lib.core.Attacks.UAP import uap_sgd_yolop

import datetime
import matplotlib.pyplot as plt

sys.path.append('/media/yunge/HDD1/REU-2024-YOLOP/pytorch-CycleGAN-and-pix2pix')  # Ensure the path is correct
from options.test_options import TestOptions

sns.set_theme(style="darkgrid", palette="Spectral")

def plot_metrics(results_df, metric_list, x_param, attack_type, baseline_metrics, exp_output_dir):
    """
    Plot metrics for the experiment results and save the plots.

    Args:
        results_df (pd.DataFrame): DataFrame containing the experiment results.
        metric_list (list): List of metrics to plot.
        x_param (str): Parameter to plot on the x-axis.
        attack_type (str): Type of attack.
        baseline_metrics (dict): Baseline metrics.
        exp_output_dir (str): Directory to save the plots.
    """
    plt.figure(figsize=(14, 10))

    for i, metric in enumerate(metric_list, 1):
        plt.subplot(3, 3, i)
        sns.lineplot(x=results_df[x_param], y=results_df[metric], marker='o', label='Experiment Results')
        plt.axhline(y=baseline_metrics[metric], color='r', linestyle='--', label='Baseline')
        plt.xlabel(x_param)
        plt.ylabel(metric)
        plt.title(f'{metric} vs {x_param} for {attack_type}')
        if i == 1:
            plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{exp_output_dir}/{attack_type}_metrics_plot.png')
    plt.close()

def plot_metrics_separate(results_df, metric_list, x_param, attack_type, baseline_metrics, exp_output_dir):
    """
    Plot metrics for the experiment results and save the plots separately for each perturbation type.

    Args:
        results_df (pd.DataFrame): DataFrame containing the experiment results.
        metric_list (list): List of metrics to plot.
        x_param (str): Parameter to plot on the x-axis.
        attack_type (str): Type of attack.
        baseline_metrics (dict): Baseline metrics.
        exp_output_dir (str): Directory to save the plots.
    """
    perturb_types = results_df['perturb_type'].unique()
    for perturb_type in perturb_types:
        plt.figure(figsize=(14, 10))
        subset = results_df[results_df['perturb_type'] == perturb_type]
        
        for i, metric in enumerate(metric_list, 1):
            plt.subplot(3, 3, i)
            sns.lineplot(x=subset[x_param], y=subset[metric], marker='o', label=f'{perturb_type} - {subset["perturb_value"].iloc[0]}')
            plt.axhline(y=baseline_metrics[metric], color='r', linestyle='--', label='Baseline')
            plt.xlabel(x_param)
            plt.ylabel(metric)
            plt.title(f'{metric} vs {x_param} for {attack_type} - {perturb_type}')
            if i == 1:
                plt.legend()
            plt.grid(True)

        plt.tight_layout()
        perturb_dir = os.path.join(exp_output_dir, perturb_type)
        os.makedirs(perturb_dir, exist_ok=True)
        plt.savefig(f'{perturb_dir}/{attack_type}_{perturb_type}_metrics_plot.png')
        plt.close()
        
def calculate_percentage_drop(initial, current):
    """
    Calculate the percentage drop from the initial value to the current value.

    Args:
        initial (float): Initial value.
        current (float): Current value.

    Returns:
        float: Percentage drop.
    """
    if initial == 0:
        return 0.0
    return ((initial - current) / initial) * 100

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(description='Test Multitask network')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default='/data2/zwt/wd/YOLOP/runs/BddDataset/detect_and_segbranch_whole/epoch-169.pth',
                        help ='model.pth path(s)')
    parser.add_argument('--conf_thres',
                        type=float,
                        default=0.001,
                        help ='object confidence threshold')
    parser.add_argument('--iou_thres',
                        type=float,
                        default=0.6,
                        help ='IOU threshold for NMS')
    
   # Adding new arguments for dataset and attack type
    parser.add_argument('--dataset',
                        type=str,
                        choices=['Carla', 'BDD100k'],
                        help ='Choice of dataset: Carla or BDD100k',
                        default='BDD100k')
    parser.add_argument('--attack',
                        type=str,
                        choices=['FGSM', 'JSMA', 'UAP', 'CCP', 'None'],
                        help ='Choice of attack: FGSM, JSMA, UAP, CCP, or None',
                        default= 'None')
    
    parser.add_argument('--experiment_mode',
                        type = int, choices = [0, 1],
                        help= 'Run with experiment mode? (1 (True): Runs with several pre-generated values. '
                          '0 (False): Provide your own parameters. FGSM: epsilon, attack type; '
                          'JSMA: num_pixels, perturbation value, attack type; '
                          'UAP: max_iterations, epsilon, delta, num_classes, targeted, batch_size; '
                          'CCP: epsilon, color_channel)',
                        default = 1)
    
    # New arguments for FGSM
    parser.add_argument('--epsilon',
                        type=float,
                        help='Epsilon value for FGSM or CCP attack',
                        default=0.1)
    parser.add_argument('--fgsm_attack_type', 
                        type=str, 
                        choices=['FGSM', 'FGSM_WITH_NOISE', 'ITERATIVE_FGSM'],
                        help ='Type of FGSM attack. Options include: FGSM, FGSM_WITH_NOISE, and ITERATIVE_FGSM',
                        default='FGSM')
    
    # New arguments for JSMA
    parser.add_argument('--num_pixels',
                        type=int,
                        help="The number of pixels to be perturbed after saliency calculation.",
                        default = 10)
    parser.add_argument('--jsma_perturbation',
                        type=float,
                        help="The number of pixels to be perturbed after saliency calculation.",
                        default = .1)
    parser.add_argument('--jsma_attack_type',
                        type = str,
                        choices = ["add", "set", "noise"],
                        help = "Select the type of perturbation to be applied to the highest scoring pixels. Options include add, set, and noise.",
                        default = "noise"
                        )
    
    # New arguments for UAP
    parser.add_argument('--uap_max_iterations',
                        type=int,
                        help='Maximum number of iterations for UAP attack',
                        default=10)
    parser.add_argument('--uap_eps',
                        type=float,
                        help='Epsilon value for UAP attack',
                        default=0.03)
    parser.add_argument('--uap_delta',
                        type=float,
                        help='Delta value for UAP attack',
                        default=0.8)
    parser.add_argument('--uap_num_classes',
                        type=int,
                        help='Number of classes for UAP attack',
                        default=None)
    parser.add_argument('--uap_targeted',
                        type=bool,
                        help='Whether the UAP attack is targeted or not',
                        default=False)
    parser.add_argument('--uap_batch_size',
                        type=int,
                        help='Batch size for UAP attack',
                        default=6)
    
    # New Args for CCP 
    parser.add_argument('--color_channel',
                    type=str,
                    choices=['R', 'G', 'B'],
                    help='Color channel to perturb (R, G, B)',
                    default='R')
    
    # Defenses 
    parser.add_argument('--resizer', type=str, help='Desired WIDTHxHEIGHT of your resized image(1024x576)')
    parser.add_argument('--quality', type=int, help='Desired quality for JPEG compression output. 0 - 100(80)')
    parser.add_argument('--border_type', type=str, choices=['default', 'constant', 'reflect', 'replicate'], help= 'border type for Gaussian Blurring(default)')
    parser.add_argument('--gauss', type=str, help="Apply Gaussian Blurring to image. Specify ksize as WIDTHxHEIGHT(3x3)")
    parser.add_argument('--noise', type=float, help='Add Gaussian Noise to image. Specify sigma value for noise generation.(0.01)')
    parser.add_argument('--bit_depth', type=int, help='Choose bit value between 1 - 8(8)')

    args = parser.parse_args()
    return args

def create_and_save_table(results_df, normal_metrics, metrics, identifier_column, output_filename_prefix, identifier_values, display_identifier, combine=False):
    """
    Create and save a table of results, including percentage drops.

    Args:
        results_df (pd.DataFrame): DataFrame containing the experiment results.
        normal_metrics (dict): Baseline metrics.
        metrics (list): List of metrics to include in the table.
        identifier_column (str): Column to identify different runs.
        output_filename_prefix (str): Prefix for the output filenames.
        identifier_values (list): List of identifier values.
        display_identifier (str): Display identifier for the normal metrics.
        combine (bool): Whether to combine all results into a single table.
    """   
    percentage_drops_df = results_df.copy()

    for metric in metrics:
        initial_value = normal_metrics[metric]
        percentage_drops_df[metric] = results_df[metric].apply(lambda x: calculate_percentage_drop(initial_value, x))

    if combine:
        display_df = results_df.copy()

        # Add normal metrics as the first row
        normal_metrics_row = normal_metrics.copy()
        normal_metrics_row[identifier_column] = display_identifier
        for metric in metrics:
            normal_metrics_row[f'{metric}_drop'] = 0
        
        normal_metrics_row_df = pd.DataFrame([normal_metrics_row])
        display_df = pd.concat([normal_metrics_row_df, display_df])

        # Round each metric to 4 significant figures
        for metric in metrics:
            display_df[metric] = display_df[metric].apply(lambda x: f'{x:.4g}')

        # Add percentage drops next to metrics
        for metric in metrics:
            display_df[f'{metric}_drop'] = percentage_drops_df[metric].apply(lambda x: f'{x:.2f}%')

        # Interleave the metric and drop columns
        interleaved_columns = []
        for metric in metrics:
            interleaved_columns.append(metric)
            interleaved_columns.append(f'{metric}_drop')

        interleaved_columns = [identifier_column] + interleaved_columns

        # Reorder the columns in display_df
        display_df = display_df[interleaved_columns]

        # Plotting the DataFrame
        fig, ax = plt.subplots(figsize=(28, 16))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')

        # Style the drop columns to be red
        for (i, j), cell in table.get_celld().items():
            if j > 0 and display_df.columns[j].endswith('_drop'):
                cell.set_text_props(color='red')

        # Increase font size
        table.auto_set_font_size(False)
        table.set_fontsize(11)

        # Save the table as an image
        plt.savefig(f'{output_filename_prefix}_combined.png', bbox_inches='tight', dpi=600)
        plt.close(fig)

    else:
        for identifier_value in identifier_values:
            display_df = results_df[results_df[identifier_column] == identifier_value].copy()

            # Add normal metrics as the first row
            normal_metrics_row = normal_metrics.copy()
            normal_metrics_row[identifier_column] = display_identifier
            for metric in metrics:
                normal_metrics_row[f'{metric}_drop'] = 0
            
            normal_metrics_row_df = pd.DataFrame([normal_metrics_row])
            display_df = pd.concat([normal_metrics_row_df, display_df])

            # Round each metric to 4 significant figures
            for metric in metrics:
                display_df[metric] = display_df[metric].apply(lambda x: f'{x:.4g}')

            # Add percentage drops next to metrics
            for metric in metrics:
                display_df[f'{metric}_drop'] = percentage_drops_df[metric].apply(lambda x: f'{x:.2f}%')

            # Interleave the metric and drop columns
            interleaved_columns = []
            for metric in metrics:
                interleaved_columns.append(metric)
                interleaved_columns.append(f'{metric}_drop')

            interleaved_columns = [identifier_column] + interleaved_columns

            # Reorder the columns in display_df
            display_df = display_df[interleaved_columns]

            # Plotting the DataFrame
            fig, ax = plt.subplots(figsize=(28, 16))
            ax.axis('tight')
            ax.axis('off')

            # Create table
            table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center')

            # Style the drop columns to be red
            for (i, j), cell in table.get_celld().items():
                if j > 0 and display_df.columns[j].endswith('_drop'):
                    cell.set_text_props(color='red')

            # Increase font size
            table.auto_set_font_size(False)
            table.set_fontsize(11)

            # Save the table as an image
            plt.savefig(f'{output_filename_prefix}_{identifier_value}.png', bbox_inches='tight', dpi=600)
            plt.close(fig)
            
def create_log_message(da_segment_results, ll_segment_results, detect_results, total_loss, times):
    """
    Create a log message for the validation results.

    Args:
        da_segment_results (tuple): Results for driving area segmentation.
        ll_segment_results (tuple): Results for lane line segmentation.
        detect_results (tuple): Results for object detection.
        total_loss (float): Total loss value.
        times (tuple): Timing information.

    Returns:
        str: Formatted log message.
    """
    return 'Test:    Loss({loss:.3f})\n' \
           'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
           'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
           'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
           'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
               loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
               ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
               p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
               t_inf=times[0], t_nms=times[1]
           )

def process_results(exp_output_dir, logger, da_segment_results, ll_segment_results, detect_results, total_loss, normal_metrics, param_name, param_value):
    """
    Process and save the validation results.

    Args:
        exp_output_dir (str): Directory to save the results.
        logger (logging.Logger): Logger to log the results.
        da_segment_results (tuple): Results for driving area segmentation.
        ll_segment_results (tuple): Results for lane line segmentation.
        detect_results (tuple): Results for object detection.
        total_loss (float): Total loss value.
        normal_metrics (dict): Baseline metrics.
        param_name (str): Name of the parameter being varied.
        param_value (float): Value of the parameter being varied.
    """
    results_df = pd.DataFrame({
        'da_acc_seg': [da_segment_results[0]],
        'da_IoU_seg': [da_segment_results[1]],
        'da_mIoU_seg': [da_segment_results[2]],
        'll_acc_seg': [ll_segment_results[0]],
        'll_IoU_seg': [ll_segment_results[1]],
        'll_mIoU_seg': [ll_segment_results[2]],
        'detect_result': [detect_results[2]],  # mAP@0.5
        'loss_avg': [total_loss],
        param_name: [param_value]
    })
    metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
    create_and_save_table(results_df, normal_metrics, metrics, param_name, f'{exp_output_dir}/{param_name}_results', results_df[param_name].unique(), '0', combine=True)
    # plot_metrics(results_df, metrics, param_name, param_name, normal_metrics, exp_output_dir)

def create_normal_metrics(da_segment_results, ll_segment_results, detect_results, total_loss):
    """
    Create a dictionary of baseline metrics.

    Args:
        da_segment_results (tuple): Results for driving area segmentation.
        ll_segment_results (tuple): Results for lane line segmentation.
        detect_results (tuple): Results for object detection.
        total_loss (float): Total loss value.

    Returns:
        dict: Dictionary of baseline metrics.
    """
    return {
        'da_acc_seg': da_segment_results[0],
        'da_IoU_seg': da_segment_results[1],
        'da_mIoU_seg': da_segment_results[2],
        'll_acc_seg': ll_segment_results[0],
        'll_IoU_seg': ll_segment_results[1],
        'll_mIoU_seg': ll_segment_results[2],
        'detect_result': detect_results[2],  # mAP@0.5
        'loss_avg': total_loss,
        'da_acc_seg_drop': 0.0,
        'da_IoU_seg_drop': 0.0,
        'da_mIoU_seg_drop': 0.0,
        'll_acc_seg_drop': 0.0,
        'll_IoU_seg_drop': 0.0,
        'll_mIoU_seg_drop': 0.0,
        'detect_result_drop': 0.0,  # mAP@0.5
        'loss_avg_drop': 0.0
    }
    
def main():
    """
    Main function to run the experiment and process the results.
    """
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)
    
    # Attack type selection based on argument
    attack_type = args.attack
    
    if attack_type == 'None':
        attack_type = None
        print("None selected. Will run only a normal validation.")
    else:
        print(f"{attack_type} selected\n")
                

    # TODO: handle distributed training logger
    # Create the base directory for the run
    base_logger, base_output_dir, base_tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test', attack_type=attack_type, epsilon=args.epsilon, channel=args.color_channel, experiment_number=1)

    # Log the configuration
    base_logger.info(cfg)
    base_logger.info(pprint.pformat(args))
    base_logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=base_tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Build up model
    print("Begin to build up model...\n")
    
    # Device selection
    device = select_device(base_logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS))
    model = get_net(cfg)
    print("Finish build model\n")
    
    # Define loss function and optimizer
    criterion = get_loss(cfg, device=device)

    # Load checkpoint model
    model_dict = model.state_dict()
    checkpoint_file = args.weights[0]
    base_logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    base_logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1
    print("Build model finished")

    print("Begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=0, # must be 0 or pickling error will be generated
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    
    print('Load data finished')
    
    epoch = 0

    startTime = datetime.datetime.now()
    
    # Baseline validation to generate normal metrics
    normal_logger, normal_output_dir = create_experiment_logger(base_output_dir, 0, 'baseline')
        
    
    da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
        epoch, cfg, valid_loader, valid_dataset, model, criterion,
        normal_output_dir, base_tb_log_dir, writer_dict=writer_dict, logger=normal_logger, device=device, rank=-1,
        attack_type=None, experiment_number=0
    )
    normal_metrics = create_normal_metrics(da_segment_results, ll_segment_results, detect_results, total_loss)

    # Log the baseline results
    msg = create_log_message(da_segment_results, ll_segment_results, detect_results, total_loss, times)
    normal_logger.info(msg)

    def save_combined_results(results_df, attack_type, exp_output_dir):
        """
        Save all results to an Excel file
        
        Args:
            results_df (pd.DataFrame): DataFrame containing all results
            attack_type (str): Attack type
            exp_output_dir (str): Output directory
        """
        # Ensure the output directory exists
        os.makedirs(exp_output_dir, exist_ok=True)
        
        # Save as an Excel file
        output_file = os.path.join(exp_output_dir, f'{attack_type}_all_results_drivable.xlsx')
        results_df.to_excel(output_file, index=False)
        print(f"All results have been saved to: {output_file}")
    
    results = []

    # Now run validations for each attack type
    if attack_type == "FGSM":
        epsilons = [0.1] if args.experiment_mode == 1 else [args.epsilon] 
        # epsilons = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] if args.experiment_mode == 1 else [args.epsilon] 
        '''.1, .3, .5, .75, 1, 3, 5, 7, 10'''
        for experiment_number, epsilon in enumerate(epsilons, start=1):
            exp_logger, exp_output_dir = create_experiment_logger(base_output_dir, experiment_number, attack_type, epsilon=epsilon)
            
            pix2pix_params = {
                'name': 'pix2pixnew',
                'model': 'test',
                'netG': 'resnet_9blocks',
                'preprocess': 'none',
                'batch_size': 2,
                'epoch': 100,
                'dataset_mode': 'single',
                'norm': 'batch',
                'checkpoints_dir': 'pytorch-CycleGAN-and-pix2pix/checkpoints'  # Set to your actual path
            }
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                exp_output_dir, base_tb_log_dir, writer_dict=writer_dict, logger=exp_logger, device=device, rank=-1,
                attack_type=args.fgsm_attack_type, epsilon=epsilon, experiment_number=experiment_number, use_pix2pix=False, pix2pix_params=pix2pix_params,
                resizer=args.resizer, quality=args.quality, bit_depth=args.bit_depth, 
                gauss=args.gauss, border_type=args.border_type, noise=args.noise)

            msg = create_log_message(da_segment_results, ll_segment_results, detect_results, total_loss, times)
            exp_logger.info(msg)

            results.append({
                "epsilon": epsilon,
                "da_acc_seg": da_segment_results[0],
                "da_IoU_seg": da_segment_results[1],
                "da_mIoU_seg": da_segment_results[2],
                "ll_acc_seg": ll_segment_results[0],
                "ll_IoU_seg": ll_segment_results[1],
                "ll_mIoU_seg": ll_segment_results[2],
                "detect_P": detect_results[0],  # Precision
                "detect_R": detect_results[1],  # Recall
                "detect_mAP50": detect_results[2],  # mAP@0.5
                "detect_mAP": detect_results[3],  # mAP@0.5:0.95
                "loss_avg": total_loss,
                "time": times
            })
            process_results(exp_output_dir, exp_logger, da_segment_results, ll_segment_results, detect_results, total_loss, normal_metrics, 'epsilon', epsilon)
        results_df = pd.DataFrame(results)
        save_combined_results(results_df, attack_type, exp_output_dir)
        # Define normal_metrics, metrics, and param_name
        metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
        param_name = 'epsilon'

        create_and_save_table(results_df, normal_metrics, metrics, param_name, f'{exp_output_dir}/{param_name}_results', results_df[param_name].unique(), '0', combine=True)
        plot_metrics(results_df, metrics, param_name, 'FGSM', normal_metrics, exp_output_dir)

    elif attack_type == "JSMA":
        perturbation_params = [
    (46080, 0.1, 'noise'), (92160, 0.1, 'noise'), (138240, 0.1, 'noise'), (184320, 0.1, 'noise'), (368640, 0.1, 'noise'), (552960, 0.1, 'noise'), (737280, 0.1, 'noise'), (921600, 0.1, 'noise'),
    (46080, 0.2, 'noise'), (92160, 0.2, 'noise'), (138240, 0.2, 'noise'), (184320, 0.2, 'noise'), (368640, 0.2, 'noise'), (552960, 0.2, 'noise'), (737280, 0.2, 'noise'), (921600, 0.2, 'noise'),
    (46080, 0.3, 'noise'), (92160, 0.3, 'noise'), (138240, 0.3, 'noise'), (184320, 0.3, 'noise'), (368640, 0.3, 'noise'), (552960, 0.3, 'noise'), (737280, 0.3, 'noise'), (921600, 0.3, 'noise'),
    (46080, 0.4, 'noise'), (92160, 0.4, 'noise'), (138240, 0.4, 'noise'), (184320, 0.4, 'noise'), (368640, 0.4, 'noise'), (552960, 0.4, 'noise'), (737280, 0.4, 'noise'), (921600, 0.4, 'noise'), 
    (46080, 0.5, 'noise'), (92160, 0.5, 'noise'), (138240, 0.5, 'noise'), (184320, 0.5, 'noise'), (368640, 0.5, 'noise'), (552960, 0.5, 'noise'), (737280, 0.5, 'noise'), (921600, 0.5, 'noise'),
    (46080, 0.6, 'noise'), (92160, 0.6, 'noise'), (138240, 0.6, 'noise'), (184320, 0.6, 'noise'), (368640, 0.6, 'noise'), (552960, 0.6, 'noise'), (737280, 0.6, 'noise'), (921600, 0.6, 'noise'),
    (46080, 0.7, 'noise'), (92160, 0.7, 'noise'), (138240, 0.7, 'noise'), (184320, 0.7, 'noise'), (368640, 0.7, 'noise'), (552960, 0.7, 'noise'), (737280, 0.7, 'noise'), (921600, 0.7, 'noise'), 
    (46080, 0.8, 'noise'), (92160, 0.8, 'noise'), (138240, 0.8, 'noise'), (184320, 0.8, 'noise'), (368640, 0.8, 'noise'), (552960, 0.8, 'noise'), (737280, 0.8, 'noise'), (921600, 0.8, 'noise'),
    (46080, 0.9, 'noise'), (92160, 0.9, 'noise'), (138240, 0.9, 'noise'), (184320, 0.9, 'noise'), (368640, 0.9, 'noise'), (552960, 0.9, 'noise'), (737280, 0.9, 'noise'), (921600, 0.9, 'noise'),
    (46080, 1, 'noise'), (92160, 1, 'noise'), (138240, 1, 'noise'), (184320, 1, 'noise'), (368640, 1, 'noise'), (552960, 1, 'noise'), (737280, 1, 'noise'), (921600, 1, 'noise')  # 921600 pixels in a 1280 x 720 image
] if args.experiment_mode else [(args.num_pixels, args.jsma_perturbation, args.jsma_attack_type)]
        
        for experiment_number, (num_pixels, perturb_value, perturb_type) in enumerate(perturbation_params, start=1):
            print(f"The number of pixels - {num_pixels}")
            print(f"The perturb value - {perturb_value}")

            saliency_maps = calculate_saliency(model, valid_loader, device, cfg, criterion)
            
            # perturbed_images, _ = find_and_perturb_highest_scoring_pixels(images, saliency_maps, num_pixels, perturb_value, perturbation_type=perturb_type)
            exp_logger, exp_output_dir = create_experiment_logger(base_output_dir, experiment_number, attack_type, epsilon=perturb_value, num_pixels=num_pixels)
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                exp_output_dir, base_tb_log_dir, writer_dict=writer_dict, logger=exp_logger, device=device, rank=-1,
                attack_type=attack_type, num_pixels=num_pixels, experiment_number=experiment_number, epsilon=perturb_value,
                perturb_type=perturb_type, saliency_maps=saliency_maps
            )
            
            msg = create_log_message(da_segment_results, ll_segment_results, detect_results, total_loss, times)
            exp_logger.info(msg)
            results.append({
                "num_pixels": num_pixels,
                "perturb_value": perturb_value,
                "perturb_type": perturb_type,
                "da_acc_seg": da_segment_results[0],
                "da_IoU_seg": da_segment_results[1],
                "da_mIoU_seg": da_segment_results[2],
                "ll_acc_seg": ll_segment_results[0],
                "ll_IoU_seg": ll_segment_results[1],
                "ll_mIoU_seg": ll_segment_results[2],
                "detect_P": detect_results[0],  # Precision
                "detect_R": detect_results[1],  # Recall
                "detect_mAP50": detect_results[2],  # mAP@0.5
                "detect_mAP": detect_results[3],  # mAP@0.5:0.95
                "loss_avg": total_loss,
                "time": times
            })
            process_results(exp_output_dir, exp_logger, da_segment_results, ll_segment_results, detect_results, total_loss, normal_metrics, 'num_pixels', num_pixels)

        results_df = pd.DataFrame(results)
        save_combined_results(results_df, attack_type, exp_output_dir)
        print(f"\n{results_df}\n")

        # Define normal_metrics, metrics, and param_name
        metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
        param_name = 'perturb_value'

        create_and_save_table(results_df, normal_metrics, metrics, param_name, f'{exp_output_dir}/{param_name}_results', results_df[param_name].unique(), '0', combine=True)
        plot_metrics(results_df, metrics, param_name, 'JSMA', normal_metrics, exp_output_dir)

    elif attack_type == "UAP":
        uap_params = [
            (5, 0, 0, None, None, 7),
            (5, 0.001, 0.0001, None, None, 7),
            (5, 0.1, 0.01, None, None, 7),
            (5, 0.5, 0.05, None, None, 7),
            (5, 1.0, 0.1, None, None, 7),
            # (10, 1.6, 0.85, None, None, 13),
            # (10, 2.1, 0.9, None, None, 14),
            # (10, 2.6, 0.95, None, None, 15),
            # (10, 3.1, 1.0, None, None, 16),
            # (10, 3.6, 1.05, None, None, 17),
            # (10, 4.1, 1.1, None, None, 18),
            # (10, 4.6, 1.15, None, None, 19),
            # (10, 5.1, 0.7, None, None, 20),
            # (10, 5.6, 0.75, None, None, 21),
            # (10, 6.1, 0.8, None, None, 22),
            # (10, 6.6, 0.85, None, None, 23),
            # (10, 7.1, 0.9, None, None, 24),
            # (10, 7.6, 0.95, None, None, 25),
            # (10, 8.1, 1.0, None, None, 26),
            # (10, 8.6, 1.05, None, None, 27),
            # (10, 9.1, 1.1, None, None, 28),
            # (10, 9.6, 1.15, None, None, 29) 
        ] if args.experiment_mode else [(args.uap_max_iterations, args.uap_eps, args.uap_delta, args.uap_num_classes, args.uap_targeted, args.uap_batch_size)]
        
        for experiment_number, (nb_epoch, eps, step_decay, y_target, layer_name, beta) in enumerate(uap_params, start=1):
            uap, loss_history = uap_sgd_yolop(model, valid_loader, device, nb_epoch, eps, criterion, step_decay, beta, y_target, None, layer_name)
            
            exp_logger, exp_output_dir = create_experiment_logger(base_output_dir, experiment_number, attack_type, epsilon=eps, step_decay=step_decay)
            
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                exp_output_dir, base_tb_log_dir, writer_dict=writer_dict, logger=exp_logger, device=device, rank=-1,
                attack_type=attack_type, step_decay=step_decay, epsilon=eps, experiment_number=experiment_number, uap=uap
            )
            msg = create_log_message(da_segment_results, ll_segment_results, detect_results, total_loss, times)
            exp_logger.info(msg)
            results.append({
                "nb_epoch": nb_epoch,
                "eps": eps,
                "step_decay": step_decay,
                "y_target": y_target,
                "layer_name": layer_name,
                "beta": beta,
                "da_acc_seg": da_segment_results[0],
                "da_IoU_seg": da_segment_results[1],
                "da_mIoU_seg": da_segment_results[2],
                "ll_acc_seg": ll_segment_results[0],
                "ll_IoU_seg": ll_segment_results[1],
                "ll_mIoU_seg": ll_segment_results[2],
                "detect_P": detect_results[0],  # Precision
                "detect_R": detect_results[1],  # Recall
                "detect_mAP50": detect_results[2],  # mAP@0.5
                "detect_mAP": detect_results[3],  # mAP@0.5:0.95
                "loss_avg": total_loss,
                "time": times
            })
            process_results(exp_output_dir, exp_logger, da_segment_results, ll_segment_results, detect_results, total_loss, normal_metrics, 'epsilon', eps)

        results_df = pd.DataFrame(results)
        save_combined_results(results_df, attack_type, exp_output_dir)
        # Define normal_metrics, metrics, and param_name
        # normal_metrics = create_normal_metrics(da_segment_results, ll_segment_results, detect_results, total_loss)
        metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
        param_name = 'eps'

        create_and_save_table(results_df, normal_metrics, metrics, param_name, f'{exp_output_dir}/{param_name}_results', results_df[param_name].unique(), '0', combine=True)
        plot_metrics(results_df, metrics, param_name, 'UAP', normal_metrics, exp_output_dir)

    elif attack_type == "CCP":
        ccp_params = [
        (0, 'R'), (0.001, 'R'), (0.01, 'R'), (0.05, 'R'), (0.10, 'R'), (0.20, 'R'), (0.50, 'R'), (1.00, 'R'),
        (0, 'G'), (0.001, 'G'), (0.01, 'G'), (0.05, 'G'), (0.10, 'G'), (0.20, 'G'), (0.50, 'G'), (1.00, 'G'),
        (0, 'B'), (0.001, 'B'), (0.01, 'B'), (0.05, 'B'), (0.10, 'B'), (0.20, 'B'), (0.50, 'B'), (1.00, 'B')
        ] if args.experiment_mode else [(args.epsilon, args.color_channel)]
        # ccp_params = [
        # (0.00, 'R'), 
        # (0.00, 'G'), 
        # (0.00, 'B')
        # ] if args.experiment_mode else [(args.epsilon, args.color_channel)]
        for experiment_number, (epsilon, color_channel) in enumerate(ccp_params, start=1):
            exp_logger, exp_output_dir = create_experiment_logger(base_output_dir, experiment_number, attack_type, epsilon=epsilon, channel=color_channel)
            
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                exp_output_dir, base_tb_log_dir, writer_dict=writer_dict, logger=exp_logger, device=device, rank=-1,
                attack_type=attack_type, epsilon=epsilon, channel=color_channel, experiment_number=experiment_number
            )
            msg = create_log_message(da_segment_results, ll_segment_results, detect_results, total_loss, times)
            exp_logger.info(msg)
            results.append({
                "epsilon": epsilon,
                "color_channel": color_channel,
                "da_acc_seg": da_segment_results[0],
                "da_IoU_seg": da_segment_results[1],
                "da_mIoU_seg": da_segment_results[2],
                "ll_acc_seg": ll_segment_results[0],
                "ll_IoU_seg": ll_segment_results[1],
                "ll_mIoU_seg": ll_segment_results[2],
                "detect_P": detect_results[0],  # Precision
                "detect_R": detect_results[1],  # Recall
                "detect_mAP50": detect_results[2],  # mAP@0.5
                "detect_mAP": detect_results[3],  # mAP@0.5:0.95
                "loss_avg": total_loss,
                "time": times
            })
            process_results(exp_output_dir, exp_logger, da_segment_results, ll_segment_results, detect_results, total_loss, normal_metrics, 'epsilon', epsilon)

        results_df = pd.DataFrame(results)
        save_combined_results(results_df, attack_type, exp_output_dir)
        # Define normal_metrics, metrics, and param_name
        # normal_metrics = create_normal_metrics(da_segment_results, ll_segment_results, detect_results, total_loss)
        metrics = ['da_acc_seg', 'da_IoU_seg', 'da_mIoU_seg', 'll_acc_seg', 'll_IoU_seg', 'll_mIoU_seg', 'loss_avg']
        param_name = 'epsilon'

        create_and_save_table(results_df, normal_metrics, metrics, param_name, f'{exp_output_dir}/{param_name}_results', results_df[param_name].unique(), '0', combine=True)

        # Plot combined metrics for each color channel
        sns.set(style="darkgrid")
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))

        for idx, metric in enumerate(metrics):
            ax = axes if len(metrics) == 1 else axes[idx]
            for color_channel in ['R', 'G', 'B']:
                channel_df = results_df[results_df['color_channel'] == color_channel]
                if color_channel == 'R':
                    sns.lineplot(data=channel_df, x=param_name, y=metric, color='red', marker='o', ax=ax, label='R')
                elif color_channel == 'G':
                    sns.lineplot(data=channel_df, x=param_name, y=metric, color='green', marker='o', ax=ax, label='G')
                elif color_channel == 'B':
                    sns.lineplot(data=channel_df, x=param_name, y=metric, color='blue', marker='o', ax=ax, label='B')
            ax.set_title(f'{metric} vs {param_name} for CCP')
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.legend(title='Color Channel')

        plt.tight_layout()
        plt.savefig(f"{exp_output_dir}/combined_metrics.png")
        plt.show()

        
    endTime = datetime.datetime.now()
    print("Test Finish")
    print("Starting time: ", startTime)
    print("Ending time: ", endTime.strftime("%Y-%m-%d %H:%M:%S"))
    
if __name__ == '__main__':
    main()