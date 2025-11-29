from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.impute import KNNImputer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR/ "SUBMISSION" / "DataFiles" / "TRAIN"

def dataframe():
    df_generics_train = pd.read_csv(DATA_DIR / "df_generics_train.csv")
    df_med_train = pd.read_csv(DATA_DIR / "df_medicine_info_train.csv")
    df_volume_train = pd.read_csv(DATA_DIR / "df_volume_train.csv")

    imputer = KNNImputer(n_neighbors=10)
    df_generics_train[["n_gxs"]] = imputer.fit_transform(df_generics_train[["n_gxs"]])
    df_med_train[["hospital_rate"]] = imputer.fit_transform(df_med_train[["hospital_rate"]])

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

    merged_df = merged_df.drop(columns=["brand_name"])

    cols = ["biological", "small_molecule"]
    merged_df[cols] = merged_df[cols].astype(float)

    merged_df = pd.get_dummies(
        merged_df,
        columns = ["country", "ther_area", "main_package"],
        drop_first = False,
        dtype = float)
    onehot_cols = [col for col in merged_df.columns if col.startswith("main_package") or col.startswith("ther_area") or col.startswith("country")]

    otras_cols = [col for col in merged_df.columns if col not in onehot_cols]

    merged_df = merged_df[onehot_cols + otras_cols]

    pd.set_option('display.max_columns', None)
    print(merged_df.head())
    #print(merged_df.shape)
    #print("N categories country:", df_med_train['country'].nunique())
    #print("N categories ther_area:", df_med_train['ther_area'].nunique())
    #print("N categories main_package:", df_med_train['main_package'].nunique())
    
    #Cargando valores de X e y
    X = merged_df.iloc[:,:126].values
    y = merged_df.iloc[:,126:].values

    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    
    return X,y


def get_dataloader(batch_size=512):
    X, y = dataframe()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def split_function(X,y):
    train, validation = train_test_split(X,y, validation_size = 0.2, random_state = 42, shuffle = True)
    return train, validation


