import glob
import torch
from config import *
from dataset import create_dataloader
from svd_transform import SVDTransform, prepare_svd_data
from gan import SVDGAN
from utils import visualize_svd_reconstruction, plot_training_losses

def init_cuda():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def find_and_merge_images():
    base_path = DATA_DIR
    target_dir = os.path.join(BASE_DIR, "data/all_images")
    os.makedirs(target_dir, exist_ok=True)
    existing = len([f for f in os.listdir(target_dir) if f.endswith(IMAGE_EXTENSION)])
    if existing >= MAX_IMAGES:
        return target_dir

    image_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith('images_'):
            for root, _, files in os.walk(item_path):
                if any(f.endswith(IMAGE_EXTENSION) for f in files):
                    image_dirs.append(root)
                    break

    total = existing
    import shutil
    for dir_path in image_dirs:
        if total >= MAX_IMAGES:
            break
        for fname in os.listdir(dir_path):
            if total >= MAX_IMAGES:
                break
            if fname.endswith(IMAGE_EXTENSION):
                src = os.path.join(dir_path, fname)
                dst = os.path.join(target_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    total += 1
    return target_dir

def main():
    init_config()
    device = init_cuda()

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    with open(os.path.join(SAMPLES_DIR, 'sample_info.txt'), 'w') as f:
        for k, v in [('MAX_IMAGES', MAX_IMAGES), ('SVD_COMPONENTS', SVD_COMPONENTS),
                     ('SVD_TRAINING_SAMPLES', SVD_TRAINING_SAMPLES), ('LATENT_DIM', LATENT_DIM),
                     ('GENERATOR_HIDDEN_DIMS', GENERATOR_HIDDEN_DIMS),
                     ('DISCRIMINATOR_HIDDEN_DIMS', DISCRIMINATOR_HIDDEN_DIMS),
                     ('DROPOUT_RATE', DROPOUT_RATE), ('LEARNING_RATE_GENERATOR', LEARNING_RATE_GENERATOR),
                     ('LEARNING_RATE_DISCRIMINATOR', LEARNING_RATE_DISCRIMINATOR),
                     ('BETA1', BETA1), ('BETA2', BETA2)]:
            f.write(f"{k} = {v}\n")

    svd_path = os.path.join(MODELS_DIR, 'svd_transform.npz')
    if os.path.exists(svd_path):
        os.remove(svd_path)
    for ckpt in glob.glob(os.path.join(MODELS_DIR, 'checkpoint_*.pth')):
        os.remove(ckpt)

    data_dir = find_and_merge_images()
    dataloader = create_dataloader(image_dir=data_dir)

    real_images = prepare_svd_data(dataloader)
    svd = SVDTransform(n_components=SVD_COMPONENTS)
    svd.fit(real_images)
    visualize_svd_reconstruction(dataloader, svd)
    svd.save(svd_path)

    gan = SVDGAN(svd_transform=svd, device=device)
    gan.train(dataloader, epochs=EPOCHS)
    plot_training_losses(gan)
    gan.save_models()
    gan.save_checkpoint()

if __name__ == "__main__":
    main()