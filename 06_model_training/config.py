DATA_DIR = "/home/islamkhanx/Documents/AAA/latent_kittens/data"
OLLAMA_URL = "http://172.19.0.6:11434"

# Train Params
MODEL_NAME = "openai/clip-vit-base-patch32"
TRAIN_DATA_PATH = f"{DATA_DIR}/interim/train_clip.jsonl"
VAL_DATA_PATH = f"{DATA_DIR}/interim/val_clip.jsonl"
BATCH_SIZE = 32
LR = 5e-6
EPOCHS = 5
DEVICE = "cuda"
