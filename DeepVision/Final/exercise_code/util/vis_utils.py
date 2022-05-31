"""Utils for visualizations in notebooks"""
import matplotlib.pyplot as plt
import torch
from math import sqrt, ceil
import numpy as np
from exercise_code.data.segmentation_dataset import label_img_to_rgb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualizer(model, test_data=None):
    num_example_imgs = 4
    plt.figure(figsize=(15, 5 * num_example_imgs))
    data = next(iter(test_data))
    img = data[0]
    target = data[1]
    inputs = img
    inputs = inputs.to(device)

    outputs = model.forward(inputs)
    
    _, preds = torch.max(outputs, 1)
    pred = preds.data.cpu()
    #pred = outputs.data.cpu()

    img, target, pred = img.numpy(), target.numpy(), pred.numpy()
        # img
    for i in range(num_example_imgs):
        plt.subplot(num_example_imgs, 3, i * 3 + 1)
        plt.axis('off')
        plt.imshow(img[i].transpose(1, 2, 0))
        if i == 0:
            plt.title("Input image")

        # target
        plt.subplot(num_example_imgs, 3, i * 3 + 2)
        plt.axis('off')
        plt.imshow(label_img_to_rgb(target[i]))
        if i == 0:
            plt.title("Target image")

        # pred
        plt.subplot(num_example_imgs, 3, i * 3 + 3)
        plt.axis('off')
        plt.imshow(label_img_to_rgb(pred[i]))
        if i == 0:
            plt.title("Prediction image")

    plt.show()

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
        x0 += W + padding
        x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid