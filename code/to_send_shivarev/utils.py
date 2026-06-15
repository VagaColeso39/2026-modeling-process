import numpy as np
import torch
import matplotlib.pyplot as plt
from config import *

def visualize_svd_reconstruction(dataloader, svd_transform, num_samples=5, save_dir=None):
    if save_dir is None:
        save_dir = PLOTS_DIR
    for batch in dataloader:
        if isinstance(batch, torch.Tensor):
            images = batch[:num_samples].cpu().numpy()
        else:
            images = np.array(batch)[:num_samples]
        break
    if len(images.shape) == 4:
        images = images.squeeze(1)
    coefficients = svd_transform.transform(images)
    reconstructed = svd_transform.inverse_transform(coefficients)
    mse = np.mean((images - reconstructed) ** 2)

    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    for i in range(num_samples):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title(f'Original {i + 1}', fontsize=10)
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i + 1}', fontsize=10)
        axes[1, i].axis('off')
    plt.suptitle(f'SVD Reconstruction Quality\nMSE: {mse:.6f}', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'svd_reconstruction{IMAGE_EXTENSION}')
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return mse

def plot_training_losses(gan, save_dir=None):
    if save_dir is None:
        save_dir = PLOTS_DIR
    epochs = range(1, len(gan.d_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gan.d_losses, 'r-', label='Critic loss')
    plt.plot(epochs, gan.g_losses, 'b-', label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('WGAN-GP training losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filepath = os.path.join(save_dir, f'training_losses{IMAGE_EXTENSION}')
    plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()