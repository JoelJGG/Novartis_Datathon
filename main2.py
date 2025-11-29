from numpy._core.multiarray import dtype
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import metric_calculation 

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR/ "SUBMISSION" / "DataFiles" / "TRAIN"

def dataframe():
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

    merged_df = merged_df.drop(columns=[merged_df.columns[1]])

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

    print(merged_df.head())
    print(merged_df.shape)
    print("N categories country:", df_med_train['country'].nunique())
    print("N categories ther_area:", df_med_train['ther_area'].nunique())
    print("N categories main_package:", df_med_train['main_package'].nunique())
    
    #Cargando valores de X e y
    X = merged_df.iloc[:,:126].values
    y = merged_df.iloc[:,126:].values

    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    
    return X,y

def get_dataloader(batch_size=503):
    X, y = dataframe()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 1-> 54 Inputs 55 -> outputs class ModelNovartis(nn.Module):
class ModelNovartis(nn.Module): 
    def __init__(self,inputs=126,outputs=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(inputs,664),
            nn.ReLU(),
            nn.Linear(664,1328),
            nn.ReLU(),
            nn.Linear(1328,864),
            nn.ReLU(),
            nn.Linear(864,432),
            nn.ReLU(),
            nn.Linear(432,outputs)
        )

    def forward(self,X):
        return self.features(X)

def train(model, epochs=5000, batch_size=503, lr=1e-3):
    # dataloader de tu función
    dataloader = get_dataloader(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        total = 0

        train_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)

        for X, y in train_bar:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = metric_calculation.my_loss_fn(pred,target)
            loss.backward()
            optimizer.step()

            # tamaño real del batch (último puede ser menor)
            bs = X.size(0)

            # acumulamos loss total ponderada por batch
            running_loss += loss.item() * bs

            # MAE como medida de “correcteza”
            batch_mae = torch.mean(torch.abs(pred - y)).item()
            running_mae += batch_mae * bs

            total += bs

            avg_loss = running_loss / total
            avg_mae = running_mae / total

            # mostramos en la barra de progreso
            train_bar.set_postfix(loss=avg_loss, mae=avg_mae)

        # resumen por época
        tqdm.write(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {running_loss/total:.4f} | "
            f"Train MAE: {running_mae/total:.4f}"
        )

    # guardar modelo
    torch.save(model.state_dict(), "novartis_model.pth")
    tqdm.write("Training complete. Model saved as novartis_model.pth")

def main():
    model = ModelNovartis()
    train(model)
#Bools a binari
#Reordenar categorias ONE HOT
#


if __name__ == "__main__":
    main()
