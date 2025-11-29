from pathlib import Path
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from tqdm import tqdm
import matplotlib as plt
from torch.utils.data import TensorDataset, DataLoader
import model
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR/ "SUBMISSION" / "DataFiles" / "TEST"

def fetch_test():
    df_generics_test = pd.read_csv(DATA_DIR / "df_generics_test.csv")
    df_med_test = pd.read_csv(DATA_DIR / "df_medicine_info_test.csv")
    df_volume_test = pd.read_csv(DATA_DIR / "df_volume_test.csv")

    gen_wide = (
        df_generics_test
        .pivot_table(index=["country", "brand_name"], 
                     columns="months_postgx", 
                     values="n_gxs")
        .add_prefix("ngxs")
        .reset_index()
    )

    vol_wide = (
        df_volume_test
        .pivot_table(index=["country", "brand_name"],
                     columns="months_postgx",
                     values="volume")
        .add_prefix("volume")
        .reset_index()
    )
    
    merged_df = (
        df_med_test
        .merge(gen_wide, on=["country", "brand_name"], how="left")
        .merge(vol_wide, on=["country", "brand_name"], how="left")
    )

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
    #print(merged_df.head())
    #print(merged_df.shape)
    #print("N categories country:", df_med_test['country'].nunique())
    #print("N categories ther_area:", df_med_test['ther_area'].nunique())
    #print("N categories main_package:", df_med_test['main_package'].nunique())
    
    #Cargando valores de X e y
    X = merged_df.iloc[:,:72].values

    X = torch.from_numpy(X.astype(np.float32))
     
    return X,countries,brands

def get_dataloader(batch_size=512):
    X,countries,brands= fetch_test()
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader,countries,brands



def test(model, output_csv="output_csv"):
    k = 0
    X,countries,brands = fetch_test()
    predictions = [[],[]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    modelo.eval()

    # Pasamos todo X por el modelo de una vez
    with torch.no_grad():
        X_device = X.to(device)
        y_pred = modelo(X_device)   # [N, 24] (asumiendo como en train)

    
    y_pred_np = y_pred.cpu().numpy()   # shape (N, 24)

    rows = []
    N, H = y_pred_np.shape  # N muestras, H horizonte (24)

    for i in range(N):
        country = countries.iloc[i]
        brand = brands.iloc[i]
        for m in range(H):  # m = mes 0..23
            rows.append([
                country,
                brand,
                m,                      # months_postgx
                float(y_pred_np[i, m])  # volume
            ])

    df_out = pd.DataFrame(
        rows,
        columns=["country", "brand_name", "months_postgx", "volume"]
    )
    df_out.to_csv(output_csv, index=False)
    print(f"CSV guardado en {output_csv}")



modelo = model.ModelNovartis()
test(modelo)
