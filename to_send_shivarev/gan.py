import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *
from models import Generator, Discriminator

class SVDGAN:
    def __init__(self, svd_transform=None, device='cpu'):
        self.device = device
        self.svd_transform = svd_transform
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=LEARNING_RATE_GENERATOR,
                                      betas=(BETA1, BETA2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE_DISCRIMINATOR,
                                      betas=(BETA1, BETA2))
        self.g_losses = []
        self.d_losses = []
        self.gp_losses = []
        self.epoch = 0
        self.lambda_gp = 30.0
        self.n_critic = 8
        self.coeff_mean = None
        self.coeff_std = None

    def normalize_coefficients(self, coefficients):
        if self.coeff_mean is None:
            self.coeff_mean = coefficients.mean(dim=0, keepdim=True)
            self.coeff_std = coefficients.std(dim=0, keepdim=True) + 1e-8
        return (coefficients - self.coeff_mean) / self.coeff_std

    def denormalize_coefficients(self, coefficients):
        if self.coeff_mean is not None:
            return coefficients * self.coeff_std + self.coeff_mean
        return coefficients

    def compute_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device).expand_as(real_data)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        d_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated, inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    def set_svd_transform(self, svd_transform):
        self.svd_transform = svd_transform

    def transform_to_svd(self, images):
        if self.svd_transform._Vt_gpu is not None:
            batch_size = images.size(0)
            flat_images = images.view(batch_size, -1) - self.svd_transform._mean_gpu
            coefficients = torch.mm(flat_images, self.svd_transform._Vt_gpu.t())
        else:
            images_np = images.cpu().numpy().squeeze(1)
            coefficients_np = self.svd_transform.transform(images_np)
            coefficients = torch.FloatTensor(coefficients_np).to(self.device)
        return self.normalize_coefficients(coefficients)

    def inverse_svd_to_image(self, coefficients):
        coefficients = self.denormalize_coefficients(coefficients)
        if self.svd_transform._Vt_gpu is not None and coefficients.is_cuda:
            flat_images = torch.mm(coefficients, self.svd_transform._Vt_gpu) + self.svd_transform._mean_gpu
            h, w = self.svd_transform.image_shape
            return flat_images.view(-1, 1, h, w)
        else:
            coefficients_np = coefficients.cpu().detach().numpy()
            images_np = self.svd_transform.inverse_transform(coefficients_np)
            return torch.FloatTensor(images_np).unsqueeze(1)

    def train(self, dataloader, epochs=EPOCHS):
        sample_batches = []
        for i, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            coeffs = self.svd_transform.transform(batch)
            sample_batches.append(coeffs)
        all_coeffs = torch.cat(sample_batches, dim=0)
        self.coeff_mean = all_coeffs.mean(dim=0, keepdim=True)
        self.coeff_std = all_coeffs.std(dim=0, keepdim=True) + 1e-8

        fixed_z = torch.randn(NUM_SAMPLES_TO_GENERATE, LATENT_DIM, device=self.device)

        for epoch in range(epochs):
            self.epoch = epoch + 1
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_gp = 0.0
            num_batches = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}/{epochs}", leave=False, ncols=100)
            for batch in progress_bar:
                batch = batch.to(self.device, non_blocking=True)
                real_data = self.transform_to_svd(batch)
                batch_size = real_data.size(0)

                d_loss = 0.0
                gp = 0.0
                for _ in range(self.n_critic):
                    self.d_optimizer.zero_grad()
                    z = torch.randn(batch_size, LATENT_DIM, device=self.device)
                    with torch.no_grad():
                        fake_data = self.generator(z)
                    d_real = self.discriminator(real_data)
                    d_fake = self.discriminator(fake_data)
                    gp = self.compute_gradient_penalty(real_data, fake_data)
                    d_loss = -torch.mean(d_real) + torch.mean(d_fake) + self.lambda_gp * gp
                    d_loss.backward()
                    self.d_optimizer.step()

                self.g_optimizer.zero_grad()
                z = torch.randn(batch_size, LATENT_DIM, device=self.device)
                fake_data = self.generator(z)
                d_fake = self.discriminator(fake_data)
                g_loss = -torch.mean(d_fake)
                g_loss.backward()
                self.g_optimizer.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                epoch_gp += gp.item()
                num_batches += 1
                progress_bar.set_postfix(D=f"{d_loss.item():.2f}", G=f"{g_loss.item():.2f}", GP=f"{gp.item():.2f}")

            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            avg_gp = epoch_gp / num_batches
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)
            self.gp_losses.append(avg_gp)

            if self.epoch % SAVE_INTERVAL == 0:
                self.save_samples(fixed_z, epoch_name=f"epoch_{self.epoch}")
            if self.epoch % CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint()

    def generate(self, num_images=NUM_SAMPLES_TO_GENERATE, z=None):
        self.generator.eval()
        with torch.no_grad():
            if z is None:
                z = torch.randn(num_images, LATENT_DIM, device=self.device)
            fake_coefficients = self.generator(z)
            fake_images = self.inverse_svd_to_image(fake_coefficients)
        self.generator.train()
        return fake_images

    def save_samples(self, fixed_z=None, epoch_name="latest", save_dir=None):
        if save_dir is None:
            save_dir = SAMPLES_DIR
        os.makedirs(save_dir, exist_ok=True)
        fake_images = self.generate(z=fixed_z)
        fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < len(fake_images):
                img = fake_images[i].squeeze().cpu().numpy()
                ax.imshow(img, cmap='gray')
                ax.axis('off')
        plt.suptitle(f'Generated X-ray Images - {epoch_name}')
        plt.tight_layout()
        filepath = os.path.join(save_dir, f'{epoch_name}{IMAGE_EXTENSION}')
        plt.savefig(filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

    def save_checkpoint(self, save_dir=None):
        if save_dir is None:
            save_dir = MODELS_DIR
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'gp_losses': self.gp_losses,
            'coeff_mean': self.coeff_mean,
            'coeff_std': self.coeff_std,
        }
        filepath = os.path.join(save_dir, f'checkpoint_epoch_{self.epoch}.pth')
        torch.save(checkpoint, filepath)

    def save_models(self, save_dir=None, suffix="final"):
        if save_dir is None:
            save_dir = MODELS_DIR
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(save_dir, f'generator_{suffix}.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, f'discriminator_{suffix}.pth'))