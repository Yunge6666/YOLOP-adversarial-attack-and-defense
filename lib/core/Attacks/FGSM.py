import torch 

# FGSM helper functions         
def fgsm_attack(image, epsilon, data_grad):
    """
    Applies the Fast Gradient Sign Method (FGSM) to perturb an image.

    FGSM is an attack that uses the gradients of the loss with respect to the input image to create a new image that maximizes the loss. This method is aimed to fool models by perturbing the original image just enough to deceive the models but not too much to be noticeable by human eyes.

    Args:
    image (torch.Tensor): The original input image.
    epsilon (float): The perturbation factor, often a small number, that scales the gradient's sign. This controls the magnitude of the perturbations applied to the image.
    data_grad (torch.Tensor): The gradients of the loss with respect to the input image. This indicates the direction to modify the pixels.

    Returns:
    torch.Tensor: The perturbed image. This image is clipped to ensure all values are within the [0,1] range, assuming the original image was also normalized to this range.

    Note:
    The function assumes that the input image and the gradients are torch tensors and that the operations are performed with PyTorch.
    """
    
    print(f"\nRunning standard FGSM attack\n")
    
    # Collect the sign of the gradients
    sign_data_grad = data_grad.sign()
    
    if(epsilon == 0):
        perturbed_image = image + epsilon
    else:
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

def fgsm_attack_with_noise(image, epsilon, data_grad):
    print(f"\nRunning FGSM attack with noise\n")

    noise = torch.randn_like(image) * epsilon
    perturbed_image = image + noise
    perturbed_image = fgsm_attack(perturbed_image, epsilon, data_grad)
    return perturbed_image

def iterative_fgsm_attack(image, epsilon, data_grad, alpha, num_iter, model, criterion, target, shapes):
    print(f"\nRunning iterative FGSM\n")

    perturbed_image = image.clone()
    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        
        det_out, da_seg_out, ll_seg_out = model(perturbed_image)
        inf_out, train_out = det_out 

        total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
        model.zero_grad()
        
        total_loss.backward()
        
        perturbed_image = perturbed_image + alpha * perturbed_image.grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

