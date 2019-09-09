#!/usr/bin/env python3

# Preprocessing and augmentation routines (used only in training).

def augment(image_noisy, image_gt):
    """
    Augments an input and ground truth image.
    image_noisy: Noisy input image
    image_gt: Ground truth image
    """
    image_noisy, image_gt = random_crop(image_noisy, image_gt)
    image_noisy, image_gt = random_flip(image_noisy, image_gt)
    image_noisy, image_gt = random_transpose(image_noisy, image_gt)

    return image_noisy, image_gt

def random_crop(image_noisy, image_gt, patch_size = 512):
    """
    Randomly crops an input and ground truth image to patch_size x patch_size.
    image_noisy: Noisy input image
    image_gt: Ground truth image
    """
    H = image_noisy.shape[1]
    W = image_noisy.shape[2]

    xx = np.random.randint(0, W - patch_size)
    yy = np.random.randint(0, H - patch_size)
    image_noisy = image_noisy[:, yy:yy + patch_size, xx:xx + patch_size, :]
    image_gt = image_gt[:, yy * 2:yy * 2 + patch_size * 2, xx * 2:xx * 2 + patch_size * 2, :]

    return image_noisy, image_gt

def random_flip(image_noisy, image_gt):
    """
    Randomly flips an input and ground truth image in horizontal and vertical directions.
    image_noisy: Noisy input image
    image_gt: Ground truth image
    """
    if np.random.randint(2, size=1)[0] == 1:
        image_noisy = np.flip(image_noisy, axis=1)
        image_gt = np.flip(image_gt, axis=1)

    if np.random.randint(2, size=1)[0] == 1:
        image_noisy = np.flip(image_noisy, axis=0)
        image_gt = np.flip(image_gt, axis=0)

    return image_noisy, image_gt


def random_transpose(image_noisy, image_gt):
    """
    Randomly transposes an input and ground truth image.
    image_noisy: Noisy input image
    image_gt: Ground truth image
    """
    if np.random.randint(2, size=1)[0] == 1:
        input_noisy = np.transpose(image_noisy, (0, 2, 1, 3))
        image_gt = np.transpose(image_gt, (0, 2, 1, 3))

    return image_noisy, image_gt
