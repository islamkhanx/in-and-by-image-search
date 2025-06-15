from transformers import CLIPProcessor, CLIPModel
import torch
from train.dataset import CLIPDataset, collate_fn
from torch.utils.data import DataLoader

from train.trainer import train
from config import TRAIN_DATA_PATH, VAL_DATA_PATH, MODEL_NAME, DEVICE, LR, BATCH_SIZE

if __name__ == "__main__":
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_dataset = CLIPDataset(TRAIN_DATA_PATH, processor)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=lambda x: collate_fn(x, processor))
    val_dataset = CLIPDataset(VAL_DATA_PATH, processor)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=lambda x: collate_fn(x, processor))

    model, processor = train(model, processor, train_dataloader, val_dataloader, optimizer, epochs=5)
    model.save_pretrained(f"models/clip-ft")
    processor.save_pretrained(f"models/clip-ft")