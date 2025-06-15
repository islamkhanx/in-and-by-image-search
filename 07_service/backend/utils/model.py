from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import torch
import numpy as np
from pickle import load

with open("models/tfidf.pkl", "rb") as m:
    tfidf: TfidfVectorizer = load(m)

with open("models/pca.pkl", "rb") as m:
    pca: PCA = load(m)


def get_tfidf_vector(texts: str) -> list[float]:
    tf_idf_vectors = tfidf.transform([texts])
    embeddings = pca.transform(tf_idf_vectors)
    norms = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
    embeddings = embeddings / norms

    return embeddings.flatten().tolist()


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad
def get_image_vector(image: Image) -> np.ndarray:
    input = processor(images=[image], return_tensors="pt")
    output = model.get_image_features(**input)

    embedding = output / output.norm(2, -1)
    return embedding.cpu().numpy().flatten()


@torch.no_grad
def get_text_vector(text: str) -> np.ndarray:
    input = processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    output = model.get_text_features(**input)
    embedding = output / output.norm(2, -1)
    return embedding.cpu().numpy().flatten()
