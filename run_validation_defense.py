import argparse
import os
import json
import pprint
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from lib.utils import DataLoaderX
from tensorboardX import SummaryWriter
import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import validate
from lib.models import get_net
from lib.utils.utils import create_logger, select_device
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

def parse_args():
    """
    Parse command line arguments.

    Returns:
        args: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run validation for different attacks and defenses")
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
                        default='weights/End-to-end.pth',
                        help='model.pth path(s)')
    parser.add_argument('--defended_images_dir',
                        help='path to the Defended Images directory',
                        type=str,
                        default='DefendedImages')
    parser.add_argument('--early_stop_threshold',
                        help='early stopping threshold for total loss',
                        type=float,
                        default= .65)  
    parser.add_argument('--batch_size',
                        help='number of combinations to process in each batch',
                        type=int,
                        default=2)
    return parser.parse_args()

def read_attacked_metrics(csv_paths):
    """
    Read and combine attacked metrics from CSV files.

    Args:
        csv_paths (list): List of paths to CSV files.

    Returns:
        combined_df (pd.DataFrame): Combined dataframe of attacked metrics.
    """
    combined_df = pd.DataFrame()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df['attack_type'] = df['attack_type'].str.upper()
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def run_validation(cfg, args, attack_params, defense_params, baseline=False):
    """
    Run the validation process for a given configuration, attack, and defense.

    Args:
        cfg: Configuration object.
        args: Parsed command line arguments.
        attack_params (dict): Parameters of the attack.
        defense_params (str): Parameters of the defense.
        baseline (bool): Whether to run baseline validation.

    Returns:
        dict: Results of the validation.
    """
    if baseline:
        cfg.defrost()
        cfg.DATASET.TEST_SET = 'val'
        cfg.freeze()
        validation_type = 'normal'
        attack_type = 'Baseline'
        defense_type = 'None'
    else:
        attack_type = attack_params['attack_type']
        defense_type = defense_params

        if defense_params is None:
            cfg.defrost()
            cfg.DATASET.TEST_SET = 'val'
            cfg.freeze()
            validation_type = 'attack'
            defense_type = 'None'
        else:
            cfg.defrost()
            cfg.DATASET.TEST_SET = f'{attack_type}/{defense_params}'
            cfg.freeze()
            validation_type = 'defense'
            
    print("updating the validation type to {validation_type}")

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'test', attack_type=attack_type, defense_type=defense_params if defense_params else 'None')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG else select_device(logger, 'cpu')
    model = get_net(cfg)
    criterion = get_loss(cfg, device=device)

    model_dict = model.state_dict()
    checkpoint_file = args.weights[0]
    logger.info(f"=> loading checkpoint '{checkpoint_file}'")
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info(f"=> loaded checkpoint '{checkpoint_file}'")

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        validation_type = validation_type
    )

    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )

    epoch = 0

    perturbed_images = None

    da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
        epoch, cfg, valid_loader, valid_dataset, model, criterion,
        output_dir=final_output_dir, tb_log_dir=tb_log_dir, writer_dict=writer_dict,
        logger=logger, device=device, perturbed_images=perturbed_images if validation_type == 'attack' else None
    )

    msg = ('Test:    Loss({loss:.3f})\n'
           'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'
           'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n'
           'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'
           'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)').format(
        loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
        ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
        p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
        t_inf=times[0], t_nms=times[1])

    logger.info(msg)
    
    return {
        'attack_type': attack_type,
        'attack_params': attack_params,
        'defense_type': defense_type,
        'total_loss': total_loss,
        'da_seg_acc': da_segment_results[0],
        'da_seg_iou': da_segment_results[1],
        'da_seg_miou': da_segment_results[2],
        'll_seg_acc': ll_segment_results[0],
        'll_seg_iou': ll_segment_results[1],
        'll_seg_miou': ll_segment_results[2],
        'p': detect_results[0],
        'r': detect_results[1],
        'map50': detect_results[2],
        'map': detect_results[3],
        't_inf': times[0],
        't_nms': times[1]
    }

def save_results(results, file_name, directory='.'):
    """
    Save results to a CSV file.

    Args:
        results (list): List of result dictionaries.
        file_name (str): Name of the CSV file.
        directory (str): Directory where the file will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    print(f"Saved results to {file_path}")

def prioritize_combinations(task_list):
    """
    Prioritize combinations of attack and defense tasks.

    Args:
        task_list (list): List of (attack_params, defense_params) tuples.

    Returns:
        prioritized_combinations (list): Prioritized list of tasks.
    """
    prioritized_combinations = []
    for attack_params, defense_params in task_list:
        attack_type = attack_params['attack_type']
        if attack_type == 'FGSM':
            if defense_params in ['resizing', 'compression', 'gaussian_blur']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        elif attack_type == 'JSMA':
            if defense_params in ['resizing', 'compression', 'gaussian_blur']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        elif attack_type == 'UAP':
            if defense_params in ['resizing', 'compression']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        elif attack_type == 'CCP':
            if defense_params in ['compression', 'gaussian_blur']:
                prioritized_combinations.insert(0, (attack_params, defense_params))
        else:
            prioritized_combinations.append((attack_params, defense_params))
    return prioritized_combinations

def plot_performance_by_attack(df, metric, title_template, y_label, filename_template):
    """
    Plot performance metrics by attack type.

    Args:
        df (pd.DataFrame): DataFrame containing the results.
        metric (str): Metric to plot.
        title_template (str): Template for the plot title.
        y_label (str): Label for the y-axis.
        filename_template (str): Template for the filename.
    """
    attack_types = df['attack_type'].unique()

    for attack_type in attack_types:
        if attack_type == 'Baseline':
            continue
        
        plt.figure(figsize=(14, 8))

        # Filter for baseline and specific attack type
        filtered_df = df[(df['attack_type'] == attack_type) | (df['attack_type'] == 'Baseline')].copy()

        # Ensure all values are treated as strings and handle missing values
        filtered_df['defense_type'] = filtered_df['defense_type'].fillna('none').astype(str)
        filtered_df['attack_type'] = filtered_df['attack_type'].astype(str)
        
        filtered_df['combination'] = filtered_df.apply(
            lambda row: 'Baseline' if row['attack_type'] == 'Baseline' else f"{row['attack_type'].upper()}" if row['defense_type'] == 'none' else f"{row['attack_type'].upper()} + {row['defense_type'].upper()}",
            axis=1
        )

        # Define the order for plotting
        order = ['Baseline'] + [f"{attack_type.upper()}"] + \
                sorted(filtered_df.loc[(filtered_df['attack_type'] == attack_type) & (filtered_df['defense_type'] != 'none'), 'combination'].unique(), key=str.lower)
                
        # Generate a distinct color palette for different combinations
        unique_combinations = filtered_df['combination'].unique()
        color_palette = sns.color_palette("tab20", len(unique_combinations))
        color_mapping = {combination: color_palette[idx % len(color_palette)] for idx, combination in enumerate(unique_combinations)}
        
        # Plot the data
        ax = sns.barplot(data=filtered_df, x='combination', y=metric, errorbar=None, order=order, palette = color_mapping)
        
        # Add metric labels above the bars
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 10),
                        textcoords = 'offset points')

        plt.title(title_template.format(attack_type.upper()), color='black')
        plt.xlabel('Combination', color='black')
        plt.ylabel(y_label, color='black')
        plt.xticks(rotation=45, color='black')
        plt.yticks(color='black')
        plt.tight_layout()
        plt.ylim(0, 1)  
        filename = filename_template.format(attack_type)
        plt.savefig(filename, facecolor='white')
        plt.show()
        plt.close()

def main():
    """
    Main function to run the validation process.
    """
    args = parse_args()
    update_config(cfg, args)

    results = []
    skipped_combinations = []
    seen_combinations = set()

    save_dir = 'results'  # Specify your desired directory here

    # Create a unique identifier for the current run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not os.path.exists(args.defended_images_dir):
        print(f"Directory does not exist: {args.defended_images_dir}")
        return

    task_list = []
    file_list = []
    
    # Total number of tasks desired
    total_tasks = 100  # Set this to the desired total number of tasks

    # Calculate the quota for each attack type
    quota_per_attack = total_tasks // 4

    # Set quotas for each attack type
    attack_quotas = {
        'FGSM': quota_per_attack,
        'CCP': quota_per_attack,
        'UAP': quota_per_attack,
        'JSMA': quota_per_attack
    }
    
    attack_counts = {key: 0 for key in attack_quotas}

    # Collect all metadata files
    for root, dirs, files in os.walk(args.defended_images_dir):
        for file in files:
            if file.endswith('_metadata.json'):
                file_list.append(os.path.realpath(os.path.join(root, file)))

    # Process metadata files with tqdm progress bar
    with tqdm(total=len(file_list), desc="Processing metadata files", unit="file") as pbar:
        while not all(count >= quota_per_attack for count in attack_counts.values()):
            for metadata_path in tqdm(file_list, desc="Processing metadata files"):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                attack_type = metadata.get('attack_type', 'unknown')
                if attack_type == 'unknown':
                    print(f"Warning: 'attack_type' not found in {metadata_path}")
                    pbar.update(1)
                    continue

                # Check if the quota for this attack type is already met
                if attack_counts[attack_type] >= attack_quotas[attack_type]:
                    pbar.update(1)
                    continue

                attack_params = {
                    'attack_type': attack_type,
                    'epsilon': metadata.get('epsilon', None),
                    'num_pixels': metadata.get('num_pixels', None),
                    'channel': metadata.get('channel', None)
                }
                defense_params = metadata.get('defense_params', 'none')
                
                combination_id = (attack_params['attack_type'], defense_params, attack_params.get('epsilon'), attack_params.get('num_pixels'), attack_params.get('channel'))

                if combination_id in seen_combinations:
                    pbar.update(1)
                    continue

                seen_combinations.add(combination_id)
                task_list.append((attack_params, defense_params))

                # Increment the count for this attack type
                attack_counts[attack_type] += 1
                print(f"Added task: {attack_params}, {defense_params}")

                # Break the loop early if all quotas are met
                if all(count >= quota_per_attack for count in attack_counts.values()):
                    break
                pbar.update(1)

            # Check for unmet quotas and print messages
            for attack, quota in attack_quotas.items():
                if attack_counts[attack] < quota:
                    print(f"Continuing to process files for attack type: {attack} (found {attack_counts[attack]} / {quota})")

    # Debug: Print out the collected task list to verify
    print("Final Task List:")
    for task in task_list:
        print(task)
    
    # Baseline validation
    print("\nRunning baseline validation")
    baseline_result = run_validation(cfg, args, attack_params={}, defense_params={}, baseline=True)
    results.append(baseline_result)
    
    # Read attacked-only metrics from CSV files
    csv_paths = [
        'csvs/CCP_epsilon_0.01_channel_B_validation_results_20240713-005540.csv',
        'csvs/CCP_epsilon_0.01_channel_G_validation_results_20240713-005313.csv',
        'csvs/CCP_epsilon_0.01_channel_R_validation_results_20240713-005042.csv',
        'csvs/CCP_epsilon_0.05_channel_B_validation_results_20240713-005650.csv',
        'csvs/CCP_epsilon_0.05_channel_G_validation_results_20240713-005430.csv',
        'csvs/CCP_epsilon_0.05_channel_R_validation_results_20240713-005156.csv',
        'csvs/FGSM_epsilon_0.01_validation_results_20240712-234907.csv',
        'csvs/FGSM_epsilon_0.1_validation_results_20240712-235201.csv',
        'csvs/FGSM_epsilon_0.05_validation_results_20240712-235030.csv',
        'csvs/FGSM_epslon_0.5_validation_results_20240712-235333.csv',
        'csvs/JSMA_epsilon_0.01_validation_results_20240713-000232.csv',
        'csvs/JSMA_epsilon_0.5_validation_results_20240713-000437.csv',
        'csvs/JSMA_epsilon_1_validation_results_20240713-000653.csv',
        'csvs/UAP_step_decay_0.01_validation_results_20240713-004025.csv',
        'csvs/UAP_step_decay_0.05_validation_results_20240713-004405.csv',
        'csvs/UAP_step_decay_1_validation_results_20240713-004732.csv']
    attacked_metrics = read_attacked_metrics(csv_paths)
    
    if not attacked_metrics.empty:
        results.extend(attacked_metrics.to_dict(orient='records'))
        print("Metrics appended.")
    
    # Use tqdm for progress 
    with tqdm(total=len(task_list), desc="Validating combinations", unit="batch") as pbar:
        for i in tqdm(range(0, len(task_list), args.batch_size), desc="Validating combinations"):
            batch = task_list[i:i + args.batch_size]
            
            for attack_params, defense_params in batch:
                print(f"\n{i} Running validation for {attack_params['attack_type']} attack with {defense_params} defense and parameters {attack_params}\n")
                defense_result = run_validation(cfg, args, attack_params, defense_params)
                results.append(defense_result)
                
                if defense_result['total_loss'] > args.early_stop_threshold:
                    print(f"Early stopping: Loss {defense_result['total_loss']} exceeds threshold {args.early_stop_threshold}")
                    skipped_combinations.append(defense_result)
                    save_results(skipped_combinations, f'skipped_combinations_{timestamp}.csv', save_dir)
                    continue

                save_results(results, f'validation_results_{timestamp}.csv', save_dir)
                print(f"Validation result: {defense_result}")
            
            pbar.update(1)
            print(f"\nthe value of i is {i}\n")
            
            if i == 40:
                print(f"Processed fourth batch, breaking now.")
                break

    if results:
        save_results(results, f'validation_results_{timestamp}.csv', save_dir)

        df_results = pd.DataFrame(results)
        df_results.to_csv(os.path.join(save_dir, f'validation_results_{timestamp}.csv'), index=False)
        print(df_results)

        # Visualize results
        metrics = ['da_seg_acc', 'da_seg_iou', 'da_seg_miou', 'll_seg_acc', 'll_seg_iou', 'll_seg_miou', 'p', 'r', 'map50', 'map']
        attack_types = df_results['attack_type'].unique()

        for metric in metrics:
            plot_performance_by_attack(df_results, metric, '{} - {}'.format(metric.upper(), metric.replace('_', ' ').title()), metric.replace('_', ' ').title(), os.path.join(save_dir, '{}_{}_performance_'.format(metric, '{}') + timestamp + '.png'))
    
    else:
        print(f"No results to show.")
        
    if skipped_combinations:
        save_results(skipped_combinations, f'skipped_combinations_{timestamp}.csv', save_dir)
        df_skipped = pd.DataFrame(skipped_combinations)
        df_skipped.to_csv(os.path.join(save_dir, f'skipped_combinations_{timestamp}.csv'), index=False)
        print("Skipped combinations due to exceeding the threshold:")
        print(df_skipped)

if __name__ == "__main__":
    main()