import numpy as np
import torch
from tqdm import tqdm
from config import *

class SVDTransform:
    def __init__(self, n_components=SVD_COMPONENTS):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None
        self.mean = None
        self.image_shape = None
        self._Vt_gpu = None
        self._mean_gpu = None

    def fit(self, images):
        n_samples, h, w = images.shape
        self.image_shape = (h, w)
        flat_images = images.reshape(n_samples, -1)
        self.mean = flat_images.mean(axis=0)
        centered = flat_images - self.mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.U = U[:, :self.n_components]
        self.S = S[:self.n_components]
        self.Vt = Vt[:self.n_components, :]
        self._move_to_gpu()
        return self

    def _move_to_gpu(self):
        if torch.cuda.is_available():
            self._Vt_gpu = torch.FloatTensor(self.Vt).cuda()
            self._mean_gpu = torch.FloatTensor(self.mean).cuda()

    def transform(self, images):
        if isinstance(images, torch.Tensor):
            if images.is_cuda and self._Vt_gpu is not None:
                batch_size = images.size(0)
                flat_images = images.view(batch_size, -1) - self._mean_gpu
                return torch.mm(flat_images, self._Vt_gpu.t())
            else:
                images = images.cpu().numpy()
        if isinstance(images, np.ndarray):
            n_samples = images.shape[0]
            flat_images = images.reshape(n_samples, -1) - self.mean
            return np.dot(flat_images, self.Vt.T)
        return None

    def inverse_transform(self, coefficients):
        if isinstance(coefficients, torch.Tensor) and coefficients.is_cuda:
            if self._Vt_gpu is not None:
                flat_images = torch.mm(coefficients, self._Vt_gpu) + self._mean_gpu
                h, w = self.image_shape
                return flat_images.view(-1, h, w)
        if isinstance(coefficients, torch.Tensor):
            coefficients = coefficients.cpu().numpy()
        if isinstance(coefficients, np.ndarray):
            flat_images = np.dot(coefficients, self.Vt) + self.mean
            h, w = self.image_shape
            return flat_images.reshape(-1, h, w)
        return None

    def save(self, filepath):
        np.savez(filepath, U=self.U, S=self.S, Vt=self.Vt, mean=self.mean,
                 n_components=self.n_components, image_shape=self.image_shape)

def prepare_svd_data(dataloader, max_samples=SVD_TRAINING_SAMPLES):
    real_images = []
    total_samples = 0
    for batch in tqdm(dataloader, desc="Сбор изображений"):
        if isinstance(batch, torch.Tensor):
            img = batch.cpu().numpy()
        else:
            img = np.array(batch)
        if len(img.shape) == 4:
            img = img.squeeze(1)
        real_images.append(img)
        total_samples += img.shape[0]
        if total_samples >= max_samples:
            break
    return np.vstack(real_images)[:max_samples]