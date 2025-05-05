import torch
import numpy as np
from tqdm import tqdm  


# JSMA Helper Functions 
def calculate_saliency(model, valid_loader, device, config, criterion):
    """
    Calculates the saliency maps for images from a validation data loader using a given model.

    Args:
        model (torch.nn.Module): The model used for computing the outputs.
        valid_loader (DataLoader): DataLoader containing the validation dataset.
        device (torch.device): The device (GPU/CPU) on which to perform computations.
        config (object): Configuration object containing runtime settings such as DEBUG mode.
        criterion (function): Loss function used to compute the error between predictions and targets.

    Returns:
        numpy.ndarray: An array of saliency maps for the first batch of images from the valid_loader.
    """
    
    model.eval()
    saliency_maps = []
    import time
    start_t = time.time()
    for batch_i, (img, target, paths, shapes) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        
        if not config.DEBUG:
            img = img.to(device, non_blocking=True)
            assign_target = []
            for tgt in target:
                assign_target.append(tgt.to(device))
            target = assign_target
            nb, _, height, width = img.shape  # batch size, channel, height, width
            
        # Enable gradient computation for the input image
        img.requires_grad = True
        
        # Forward pass
        det_out, da_seg_out, ll_seg_out = model(img)
        inf_out, train_out = det_out 
        
        # Calculate loss
        total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
        total_loss.backward()
        
        # Compute saliency map for each image in the batch
        saliency_maps_batch = img.grad.abs().cpu().detach().numpy()
        saliency_maps.append(saliency_maps_batch)
        
        
    end_t = time.time()
    
    print("total time in calculating saliency maps {}s".format(end_t - start_t))
    saliency_maps = np.concatenate(saliency_maps, axis=0)
    
    # Plot and save the saliency maps and original images
    img = img.detach().cpu().numpy()  # Move the original images back to CPU and convert to numpy
  
    return saliency_maps

def find_and_perturb_highest_scoring_pixels(images, saliency_maps, num_pixels_to_perturb, perturbation_value, perturbation_type='add'):
    """
    Perturbs the highest scoring pixels in images based on their saliency maps to investigate the effect on model predictions.

    Args:
        images (list of numpy.ndarray): The original images.
        saliency_maps (list of numpy.ndarray): The saliency maps of the images.
        num_pixels_to_perturb (int): Number of top pixels to perturb in each map.
        perturbation_value (float): The value to add to the top pixels in the images.
        perturbation_type (str): Type of perturbation ('add', 'set', 'noise').

    Returns:
        tuple: Contains the perturbed images tensor and the coordinates of the perturbed pixels.
    """
    perturbed_images = []
    all_top_coords = []
    for image, saliency_map in zip(images, saliency_maps):
        # Flatten the saliency map to get the indices of the top pixels
        flat_indices = np.argsort(saliency_map.flatten())[::-1]
        top_indices = flat_indices[:num_pixels_to_perturb]

        # Convert the flat indices back to 2D coordinates (y, x)
        top_coords = np.unravel_index(top_indices, saliency_map.shape)
        y_coords, x_coords = top_coords[1], top_coords[2]

        # Create a copy of the image to perturb
        perturbed_image = image.copy()
        # Apply perturbation to the top pixels
        for coord in zip(*top_coords):
            if perturbation_type == 'add':
                perturbed_image[coord] += perturbation_value
            elif perturbation_type == 'set':
                perturbed_image[coord] = perturbation_value
            elif perturbation_type == 'noise':
                perturbed_image[coord] += np.random.normal(perturbation_value)


        # Convert the perturbed image to a tensor and add a batch dimension
        perturbed_image_tensor = torch.tensor(perturbed_image, dtype=torch.float32)
        # Make sure the tensor is on the GPU if available
        if torch.cuda.is_available():
            perturbed_image_tensor = perturbed_image_tensor.cuda()
        perturbed_image_tensor = perturbed_image_tensor.unsqueeze(0)
        perturbed_images.append(perturbed_image_tensor)

        all_top_coords.append(top_coords)
    return torch.cat(perturbed_images, dim=0), all_top_coords