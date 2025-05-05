import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

def uap_sgd_yolop(model, valid_loader, device, nb_epoch, eps, criterion, step_decay, beta=12, y_target=None, loss_fn=None, layer_name=None, uap_init=None):
    """
    Universal Adversarial Perturbation (UAP) via Stochastic Gradient Descent (SGD) for YOLOP

    Args:
        model (torch.nn.Module): The YOLOP model.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (GPU/CPU) on which to perform computations.
        nb_epoch (int): Number of optimization epochs.
        eps (float): Maximum perturbation value (L-infinity norm).
        beta (float, optional): Clamping value. Default is 12.
        step_decay (float, optional): Decay rate for the step size. Default is 0.8.
        y_target (int, optional): Target class label for Targeted UAP variation. Default is None.
        loss_fn (callable, optional): Custom loss function (default is CrossEntropyLoss).
        layer_name (str, optional): Target layer name for layer maximization attack. Default is None.
        uap_init (torch.Tensor, optional): Custom perturbation to start from (default is random vector with pixel values {-eps, eps}).

    Returns:
        torch.Tensor: Adversarial perturbation.
        list: Losses per iteration.
    """
    
    if uap_init is None:
        uap = torch.rand((1, 3, 384, 640), device=device) * 2 * eps - eps  # Initialize UAP within [-eps, eps]
    else:
        uap = uap_init.to(device)
        
    uap.requires_grad = True
    optimizer = torch.optim.SGD([uap], lr=eps * step_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=step_decay)
    
    losses = []
    
    model.eval()  # Set model to evaluation mode

    print(f"The nb epochs is {nb_epoch}")
    
    for epoch in range(nb_epoch):
        epoch_loss = 0.0
        for batch_i, (img, target, paths, shapes) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            
            img = img.to(device, non_blocking=True)
            target = [tgt.to(device) for tgt in target]
            
            # Apply perturbation                  
            perturbed_img = img + uap
            
            # Forward pass
            det_out, da_seg_out, ll_seg_out = model(perturbed_img)
            inf_out, train_out = det_out 
            
            total_loss, head_losses = criterion((train_out, da_seg_out, ll_seg_out), target, shapes, model)
                    
            epoch_loss += total_loss.item()
            
            model.zero_grad()
            
            total_loss.backward()
                                    
            # Update perturbation
            optimizer.step()
            optimizer.zero_grad()
            
            # Clip perturbation to be within [-eps, eps]
            uap.data = torch.clamp(uap.data, -eps, eps)
            
        
        scheduler.step()
        losses.append(epoch_loss / len(valid_loader))

    return uap.detach(), losses
