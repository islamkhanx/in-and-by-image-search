from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel


def train_epoch(
        model: CLIPModel,
        loader,
        optimizer,
        loss_fn = nn.CosineEmbeddingLoss(margin=0.2),
        device: str = "cpu"
) -> float:
    model.train()
    total_loss = 0

    for batch in tqdm(loader):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs)
        loss = loss_fn(
            F.normalize(outputs.image_embeds, p=2, dim=1),
            F.normalize(outputs.text_embeds, p=2, dim=1),
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def val_epoch(
    model: CLIPModel,
    loader,
    optimizer,
    loss_fn = nn.CosineEmbeddingLoss(margin=0.2),
    device: str = "cpu"
):
    model.eval()
    total_loss = 0

    for batch in tqdm(loader):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs)
        loss = loss_fn(
            F.normalize(outputs.image_embeds, p=2, dim=1),
            F.normalize(outputs.text_embeds, p=2, dim=1),
            labels
        )
        total_loss += loss.item()

    return total_loss / len(loader)


def train(model, processor, train_loader, val_loader, optimizer, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer)
        print(f"Train Loss: {train_loss:.4f}")


        val_loss = val_epoch(model, val_loader)
        print(f"Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), f"models/clip_finetuned_epoch{epoch}.pt")

    return model, processor