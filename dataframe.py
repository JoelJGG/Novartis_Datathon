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
    rows = []

    #Lo siento mucho

    rows = []

    for idx, row in merged_df.iterrows():
        # Suma de volúmenes -12 a -1
        avg = (
            row["volume-12"] +
            row["volume-11"] +
            row["volume-10"] +
            row["volume-9"] +
            row["volume-8"] +
            row["volume-7"] +
            row["volume-6"] +
            row["volume-5"] +
            row["volume-4"] +
            row["volume-3"] +
            row["volume-2"] +
            row["volume-1"]
        )

        # Suma de volúmenes 0 a 24
        mean_erosion = (
            row["volume0"]  + row["volume1"]  + row["volume2"]  + row["volume3"]  +
            row["volume4"]  + row["volume5"]  + row["volume6"]  + row["volume7"]  +
            row["volume8"]  + row["volume9"]  + row["volume10"] + row["volume11"] +
            row["volume12"] + row["volume13"] + row["volume14"] + row["volume15"] +
            row["volume16"] + row["volume17"] + row["volume18"] + row["volume19"] +
            row["volume20"] + row["volume21"] + row["volume22"] + row["volume23"] 
        )

        # OJO: aquí tú haces mean_erosion / (24*avg)
        mean_erosion = mean_erosion / (24 * avg)

        if mean_erosion <= 0.25:
            b = 1 
        else:
            b = 2

        rows.append({
            "country": row["country"],
            "brand_name": row["brand_name"],
            "avg_vol": avg,
            "bucket": b
        })

    df_aux = pd.DataFrame(rows)



    brands_train = merged_df["brand_name"]
    print(brands_train.head())

    countries = merged_df["country"]
    brands = merged_df["brand_name"]

    merged_df = merged_df.drop(columns=["country","brand_name"])


    cols = ["biological", "small_molecule"]
    merged_df[cols] = merged_df[cols].astype(float)

    merged_df = pd.get_dummies(
        merged_df,
        columns = ["ther_area", "main_package"],
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
    
    return X,y,countries,brands,df_aux


def get_dataloader(batch_size=512):
    X, y,countries,brands,df_aux = dataframe()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader,countries,brands,df_aux

def split_function(X,y):
    train, validation = train_test_split(X,y, validation_size = 0.1, random_state = 42, shuffle = True)
    return train, validation

