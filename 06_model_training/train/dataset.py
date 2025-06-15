from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPProcessor
import json


class CLIPDataset(Dataset):
    def __init__(self, data_path, processor: CLIPProcessor):
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                try:
                    Image.open(data["image_path"]).convert("RGB")
                    self.data.append(json.loads(line))
                except:
                    pass
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = item["text"]
        label = item["label"]
        return image, text, label

def collate_fn(batch, processor: CLIPProcessor):
    images, texts, labels = zip(*batch)
    inputs = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    labels = torch.tensor(labels, dtype=torch.long)
    return inputs, labels