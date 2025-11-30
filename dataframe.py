from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.impute import KNNImputer
import xgboost as xgb   # 👈 añadido

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "SUBMISSION" / "DataFiles" / "TRAIN"

def dataframe():
    # -------------------------------
    # Carga de datos
    # -------------------------------
    df_generics_train = pd.read_csv(DATA_DIR / "df_generics_train.csv")
    df_med_train = pd.read_csv(DATA_DIR / "df_medicine_info_train.csv")
    df_volume_train = pd.read_csv(DATA_DIR / "df_volume_train.csv")

    # Imputación de valores faltantes
    imputer = KNNImputer(n_neighbors=10)
    df_generics_train[["n_gxs"]] = imputer.fit_transform(df_generics_train[["n_gxs"]])
    df_med_train[["hospital_rate"]] = imputer.fit_transform(df_med_train[["hospital_rate"]])

    # Pivot tables
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

    # -------------------------------
    # Feature auxiliar: bucket por mean_erosion
    # -------------------------------
    rows = []
    for idx, row in merged_df.iterrows():
        avg = sum([row[f"volume-{i}"] for i in range(1,13)])  # volúmenes -12 a -1
        mean_erosion = sum([row[f"volume{i}"] for i in range(24)]) / (24 * avg)
        b = 1 if mean_erosion <= 0.25 else 2
        rows.append({
            "country": row["country"],
            "brand_name": row["brand_name"],
            "avg_vol": avg,
            "bucket": b
        })
    df_aux = pd.DataFrame(rows)

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

    # -------------------------------
    # Enriquecimiento con XGBoost
    # -------------------------------
    X_base = merged_df.iloc[:,:72].values
    y_base = merged_df.iloc[:,72:].values

    # Modelo rápido de XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5
        )
    xgb_model.fit(X_base, y_base[:,0])  # ejemplo: primera columna de y

    # Añadimos predicciones como nueva columna
    merged_df["xgb_pred"] = xgb_model.predict(X_base)

    # Reordenamos columnas: originales primero, enriquecidas al final
    original_cols = [col for col in merged_df.columns if col != "xgb_pred"]
    enriched_cols = ["xgb_pred"]
    merged_df = merged_df[original_cols + enriched_cols]

    # -------------------------------
    # Conversión a tensores
    # -------------------------------
    X = merged_df.iloc[:,:].values
    y = y_base  # mantenemos y original

    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    
    return X,y,countries,brands,df_aux

def get_dataloader(batch_size=256):
    X, y,countries,brands,df_aux = dataframe()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader,countries,brands,df_aux

def split_function(X,y):
    train, validation = train_test_split(X,y, test_size = 0.1, random_state = 42, shuffle = True)
    return train, validation
