
import pickle
import glob
import os
import torch
import random


def load_latents(latent_path):
    r"""
    Simple utility to save latents to speed up ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps


def drop_text_condition(text_embed, im, empty_text_embed, text_drop_prob):
    if text_drop_prob > 0:
        text_drop_mask = torch.zeros((im.shape[0]), device=im.device).float().uniform_(0,
                                                                                       1) < text_drop_prob
        assert empty_text_embed is not None, ("Text Conditioning required as well as"
                                        " text dropping but empty text representation not created")
        text_embed[text_drop_mask, :, :] = empty_text_embed[0]
    return text_embed


def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,
                                                                                        1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition


def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,
                                                                                           1) > class_drop_prob
        return class_condition * class_drop_mask
    else:
        return class_condition


def generate_random_avoid_list(num_classes, current_class, max_avoid=3):
    """
    Generate a random avoid list for a single sample.
    Ensures the current class is not in the avoid list.
    
    Args:
        num_classes: Total number of classes
        current_class: The class of the current sample (to exclude from avoid list)
        max_avoid: Maximum number of classes to avoid (0 to max_avoid)
    
    Returns:
        List of class indices to avoid
    """
    # Randomly decide how many classes to avoid (0 to max_avoid)
    num_avoid = random.randint(0, max_avoid)
    
    if num_avoid == 0:
        return []
    
    # Get all classes except the current one
    available_classes = [c for c in range(num_classes) if c != current_class]
    
    # Randomly sample from available classes
    avoid_list = random.sample(available_classes, min(num_avoid, len(available_classes)))
    
    return avoid_list


def create_avoid_condition_tensor(avoid_lists, num_classes, device):
    """
    Create avoid condition tensor from a list of avoid lists.
    
    Args:
        avoid_lists: List of lists, each containing class indices to avoid
        num_classes: Total number of classes
        device: torch device
    
    Returns:
        Tensor of shape (batch_size, num_classes) with 1s at positions to avoid
    """
    batch_size = len(avoid_lists)
    avoid_tensor = torch.zeros((batch_size, num_classes), device=device)
    
    for i, avoid_list in enumerate(avoid_lists):
        for cls in avoid_list:
            avoid_tensor[i, cls] = 1.0
    
    return avoid_tensor


def drop_avoid_condition(avoid_condition, avoid_drop_prob, im):
    """
    Randomly drop avoid condition (set to zeros) for classifier-free guidance training.
    
    Args:
        avoid_condition: Tensor of shape (batch_size, num_classes)
        avoid_drop_prob: Probability of dropping the avoid condition
        im: Input image tensor (used for batch size)
    
    Returns:
        Avoid condition tensor with some rows zeroed out
    """
    if avoid_drop_prob > 0:
        avoid_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,
                                                                                           1) > avoid_drop_prob
        return avoid_condition * avoid_drop_mask
    else:
        return avoid_condition


def get_exclusion_condition(num_classes, avoid_list, batch_size, device):
    """
    Creates a soft-label vector where probability is distributed 
    among all classes EXCEPT the ones in avoid_list.
    (Legacy function - kept for backward compatibility)
    """
    # Create a base vector of ones
    target = torch.ones((batch_size, num_classes), device=device)
    
    # Set the columns for the avoided classes to 0
    if avoid_list:
        target[:, avoid_list] = 0
        
    # Normalize rows so they sum to 1 (Probability Distribution)
    # Adding epsilon to avoid division by zero if all classes were excluded
    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
    
    return target