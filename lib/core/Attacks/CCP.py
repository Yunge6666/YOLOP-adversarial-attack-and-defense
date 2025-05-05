import torch



def color_channel_perturbation(image, epsilon, data_grad, channel='R'):
    """
    Applies a perturbation to a specific color channel of an image.

    Args:
    image (torch.Tensor): The original input image.
    epsilon (float): The perturbation factor that scales the gradient's sign.
    data_grad (torch.Tensor): The gradients of the loss with respect to the input image.
    channel (str): The color channel to perturb ('R', 'G', 'B').

    Returns:
    torch.Tensor: The perturbed image with modifications in the specified color channel.
    """

    # Collect the sign of the gradients
    sign_data_grad = data_grad.sign()

    # Create a mask for the selected channel
    channel_idx = {'R': 0, 'G': 1, 'B': 2}[channel]
    mask = torch.zeros_like(image)
    mask[:, channel_idx, :, :] = 1

    # Apply the perturbation to the selected color channel
    perturbed_image = image + epsilon * sign_data_grad * mask


    return perturbed_image

