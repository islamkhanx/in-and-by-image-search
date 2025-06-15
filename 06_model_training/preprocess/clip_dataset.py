import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from config import DATA_DIR


def get_vlm_pairs(df: pl.DataFrame) -> pl.DataFrame:
    description_df = pl.read_parquet(
        os.path.join(DATA_DIR, "interim/items_description.pq")
    ).rename({
        "vlm_description": "text"
    }).join(
        df.select("item_ext_id", "image_path"),
        on="item_ext_id"
    ).with_columns(
        pl.col("text").str.split(".")
    ).explode("text").with_columns(
        pl.col("text").str.split(",")
    ).explode("text")

    return description_df

def get_item_description_pairs(df: pl.DataFrame) -> pl.DataFrame:
    pair_df = df.select(
        pl.col("image_path"),
        pl.col("item_ext_id"),
        pl.concat_str(
            pl.col("title"), pl.lit(" "),
            pl.col("brand"), pl.lit(" "),
            pl.col("cvet")
        ).alias("text")
    )
    return pair_df

def generate_training_data(pair_df: pl.DataFrame) -> list[dict]:
    training_data = []

    for row in tqdm(pair_df.iter_rows(named=True), total=len(pair_df)):
        item_id = row["item_ext_id"]

        positive_text = row["text"]
        training_data.append({
            "item_ext_id": item_id,
            "image_path": row["image_path"],
            "text": positive_text,
            "label": 1
        })
        other_texts = pair_df.filter(pl.col("item_ext_id") != item_id).sample(3)

        for neg_text in other_texts.iter_rows(named=True):
            training_data.append({
                "item_ext_id": neg_text["item_ext_id"],
                "image_path": neg_text["image_path"],
                "text": neg_text["text"],
                "label": 0
            })
    return training_data



if __name__ == "__main__":
    df = pl.read_parquet(os.path.join(DATA_DIR, "train_dataset.pq"))
    df_from_vlm = get_vlm_pairs(df)
    df_from_descriptions = get_item_description_pairs(df)
    pair_df = pl.concat((
        df_from_descriptions
            .select("image_path", "item_ext_id", "text")
            .sample(10_000, seed=1),
        df_from_vlm
            .select("image_path", "item_ext_id", "text")
    ))  # FIXME

    training_data = generate_training_data(pair_df)
    df_train, df_test = train_test_split(training_data, random_state=0)

    with open(os.path.join(DATA_DIR, "interim/train_clip.jsonl", "w")) as f:
        for item in df_train:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(DATA_DIR, "data/interim/val_clip.jsonl", "w")) as f:
        for item in df_test:
            f.write(json.dumps(item) + "\n")
