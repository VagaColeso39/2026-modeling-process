import os

BASE_DIR = "D:/xray_gan_project"
DATA_DIR = "D:/kaggle_cache/datasets/nih-chest-xrays/data/versions/3"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SAMPLES_DIR = os.path.join(OUTPUT_DIR, "generated_samples")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2
IMAGE_EXTENSION = '.png'

MAX_IMAGES = 5000
SVD_TRAINING_SAMPLES = 3000
SVD_COMPONENTS = 500

LATENT_DIM = 128
GENERATOR_HIDDEN_DIMS = [512, 1024, 2048, 1024, 512]
DISCRIMINATOR_HIDDEN_DIMS = [1024, 2048, 1024, 512]
DROPOUT_RATE = 0.2

EPOCHS = 150
LEARNING_RATE_GENERATOR = 0.0001
LEARNING_RATE_DISCRIMINATOR = 0.00005
BETA1 = 0.0
BETA2 = 0.9
SAVE_INTERVAL = 10
CHECKPOINT_INTERVAL = 10

NUM_SAMPLES_TO_GENERATE = 16
GRID_ROWS = 4
GRID_COLS = 4
PLOT_DPI = 150

USE_CUDA = True

_initialized = False

def init_config():
    global _initialized
    if _initialized:
        return
    for directory in [BASE_DIR, OUTPUT_DIR, MODELS_DIR, SAMPLES_DIR, LOGS_DIR, PLOTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    _initialized = True