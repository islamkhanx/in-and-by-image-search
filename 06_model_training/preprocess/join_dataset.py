import polars as pl
import os
from config.py import DATA_DIR


def get_item_dataset() -> pl.DataFrame:
    items = pl.read_csv(
        os.path.join(DATA_DIR, "items.csv")
    ).drop("Microcat_id", "item_id", "")
    items.glimpse()
    return items

def get_image_dataset() -> pl.DataFrame:
    images = pl.read_csv(
        os.path.join(DATA_DIR, "images.csv")
    ).drop(
        "ItemImage_id", "image_secret",
        "image_schema", "", "Image_id", "Item_id"
    )
    images.glimpse()
    return images

def get_paths_dataset() -> pl.DataFrame:
    paths = pl.read_csv(
        os.path.join(DATA_DIR, "df_path.csv")
    ).drop(
        "image_path", "", "image_ext_id"
    )
    paths = paths.with_columns(
        pl.col("image_path").str.replace(r"/home/jovyan", DATA_DIR)
    ).select("image_ext_id", "image_path")
    paths.glimpse()
    return paths

def train_test_split(df: pl.DataFrame) -> None:
    ids = df["item_ext_id"].unique()
    print((ids % 10).value_counts(normalize=True))

    df_train = df.filter(pl.col("item_ext_id") % 10 == 0)
    df_test = df.filter(pl.col("item_ext_id") % 10 != 0)

    print("Train Dataset Size", df_train.shape)
    print("Test Dataset Size", df_test.shape)

    df_train.write_parquet(os.path.join(DATA_DIR, "train_dataset.pq"))
    df_test.write_parquet(os.path.join(DATA_DIR, "test_dataset.pq"))
    print("Train and Test Dataset Saved")

if __name__ == "__main__":
    items = get_item_dataset()
    images = get_image_dataset()
    paths = get_paths_dataset()

    df = images.join(paths, on="image_ext_id")
    df = df.join(items, left_on="item_ext_id", right_on="external_id")
    print(df.head())

    df.write_parquet(
        os.path.join(DATA_DIR, "dataset.pq")
    )
    train_test_split(df)