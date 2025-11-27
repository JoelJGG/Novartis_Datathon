import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR/ "SUBMISSION" / "DataFiles" / "TRAIN"

# 1-> 54 Inputs 55 -> outputs
#class ModelNovartis():



def main():
    df_generics_train = pd.read_csv(DATA_DIR / "df_generics_train.csv")
    df_med_train = pd.read_csv(DATA_DIR / "df_medicine_info_train.csv")
    df_volume_train = pd.read_csv(DATA_DIR / "df_volume_train.csv")

    gen_wide = (
        df_generics_train
        .pivot_table(index=["country", "brand_name"], 
                     columns="months_postgx", 
                     values="n_gxs")
        .add_prefix("ngxs")
        .reset_index()
    )

    vol_wide = (
        df_volume_train
        .pivot_table(index=["country", "brand_name"],
                     columns="months_postgx",
                     values="volume")
        .add_prefix("volume")
        .reset_index()
    )

    merged_df = (
        df_med_train
        .merge(gen_wide, on=["country", "brand_name"], how="left")
        .merge(vol_wide, on=["country", "brand_name"], how="left")
    )

    merged_df = pd.get_dummies(
        merged_df,
        columns = ["country", "ther_area", "main_package"],
        drop_first = False
    )
    onehot_cols = [col for col in merged_df.columns if col.startswith("main_package") or col.startswith("ther_area") or col.startswith("country")]

    otras_cols = [col for col in merged_df.columns if col not in onehot_cols]

    merged_df = merged_df[onehot_cols + otras_cols]
    print(merged_df.head())
    print(merged_df.shape)
    print("N categories country:", df_med_train['country'].nunique())
    print("N categories ther_area:", df_med_train['ther_area'].nunique())
    print("N categories main_package:", df_med_train['main_package'].nunique())

#Bools a binari
#Reordenar categorias ONE HOT
#


if __name__ == "__main__":
    main()
