import os
from pathlib import Path
import polars as pl
import cv2
import requests
import base64
import json
import tqdm

from config import OLLAMA_URL, DATA_DIR

def llm_image_description_multiple(encoded_images: pl.Series):
    url = f"{OLLAMA_URL}/api/generate/"
    encoded_images = encoded_images.to_list()
    payload = json.dumps({
        "model": "gemma3:latest",
        "images": encoded_images,
        "prompt": (
            "Generate a detailed and structured description of the woman's clothing, focusing on key attributes "
            "such as color, type (e.g., shirt, pants, dress), size, and any visible patterns or logos."
            "Use clear, concise terms suitable for text retrieval. "
            "Example: 'blue t-shirt, cotton, 'Hello Kitty' logo, floral pattern, size M.' "
            "If there are multiple items include 'MULTIPLE' in description unless this multiple items are set or suite"
            "Avoid any non-essential text. "
            "Return only the description in Russian."
        ),
        "stream": False
    })
    try:
        response = requests.request("POST", url, data=payload)
        return response.json()["response"]
    except:
        return ""


def path_to_bytes(sample_path: Path) -> bytes:
    image = cv2.imread(str(sample_path))
    _, encoded_image = cv2.imencode(".jpeg", image)
    return encoded_image.tobytes()

def generate_item_desctiptions(df: pl.DataFrame) -> pl.DataFrame:
    df_descriptions = df \
        .group_by("item_ext_id") \
        .agg(["encoded_images"])

    vlm_descriptions = []
    encoded_images = df_descriptions["encoded_images"].to_list()

    for i in tqdm.tqdm(encoded_images):
        descr = llm_image_description_multiple(pl.Series(i))
        vlm_descriptions.append(descr)

    df_descriptions = df_descriptions.with_columns(
        vlm_description=pl.Series(vlm_descriptions)
    ).drop("encoded_images")

    return df_descriptions

def load_image_bytes(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        encoded_images=pl.col("image_path").map_elements(
            path_to_bytes, return_dtype=pl.Binary
        ).map_elements(
            lambda image_bytes: base64.b64encode(image_bytes).decode('utf-8'),
            return_dtype=pl.String
        )
    )
    return df

if __name__ == "__main__":

    df = pl.read_parquet(os.path.join(OLLAMA_URL, "train_dataset.pq"))
    df_sample = df.filter(pl.col("item_ext_id") % 300 == 0)  # FIXME
    df_sample = load_image_bytes(df_sample)

    df_descriptions = generate_item_desctiptions(df_sample)
    df_descriptions.glimpse()
    df_descriptions.write_parquet(
        os.path.join(DATA_DIR, "interim/items_description.pq")
    )