import metric_calculation
from tqdm import tqdm
import model

import dataframe
import torch.nn as nn
import torch.optim as optim
import torch

def train(modelo, epochs=5000, batch_size=126, lr=1e-3):
    # dataloader de tu función
    dataloader,countries,brands,df_aux = dataframe.get_dataloader(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)

    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print("HOOOOOOLAAAAAAAAAA")


    for epoch in range(epochs):
        modelo.train()
        running_loss = 0.0
        running_mae = 0.0
        total = 0
        train_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        print("dataloader")
        print(dataloader)
        predictions = [["country","brand_name","months_postgx","volume"]]
        print("predictions")

        for k,(X, y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = modelo.forward(X)
            
            for i in range(24):
                predictions.append([countries[k],brands[k],str(i),pred[i]])
                #country,brand,[i],vols[i]

            #Mean generic erosion (bucket)

            print(predictions)
            loss = metric_calculation.compute_metric1(y,pred,df_aux)
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
    torch.save(modelo.state_dict(), "novartis_model.pth")
    tqdm.write("Training complete. Model saved as novartis_model.pth")

modelo_para_train = model.ModelNovartis()
train(modelo_para_train)
