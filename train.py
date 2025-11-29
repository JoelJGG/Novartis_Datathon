

import metric_calculation
from tqdm import tqdm
import model

import dataframe
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score

def train(modelo, epochs=1000, batch_size=128, lr=1e-3):
    # dataloader de tu función
    dataloader,countries,brands,df_aux = dataframe.get_dataloader(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)

    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        modelo.train()
        running_loss = 0.0
        running_mae = 0.0
        total = 0
        train_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        #print("dataloader")
        #print(dataloader)
        predictions = [["country","brand_name","months_postgx","volume"]]
        y_metric = [["country","brand_name","months_postgx","volume"]]
        #print("predictions")

        for k,(X, y) in enumerate(dataloader):
            #print("dentro del for")
            #print(y)
            X, y = X.to(device), y.to(device)


            optimizer.zero_grad()
            pred = modelo.forward(X)
            #print("y")
            #print(y.detach().cpu().numpy())


            #print("pred")
            #print(pred.detach().cpu().numpy())

            tolerance = 0.10  # 10%
            diff = torch.abs(pred - y)
            allowed = tolerance * torch.abs(y)
            correct = (diff < allowed).float().sum().item()
            total_vals = y.numel()

            accuracy = correct / total_vals




            #print("pred")
            #print(pred)
            #print(len(pred[0]))
            #print(len(pred))
            bs = X.size(0)
            for i in range(bs):
                for j in range(24):
                    pred_metric = pred[i].detach().cpu().numpy()
                    predictions.append([countries[k],brands[k],str(j),pred_metric[j]])
                    y_metric.append([countries[k],brands[k],str(j),y[j].detach().cpu().numpy()])
                    #country,brand,[i],vols[i]

                #Mean generic erosion (bucket)
            #print(predictions)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()

            # tamaño real del batch (último puede ser menor)
            bs = X.size(0)

            # acumulamos loss total ponderada por batch
            running_loss += loss.item() * bs

            # MAE como medida de “correcteza”
            batch_mae = torch.mean(torch.abs(pred - y)).item()
            #print(batch_mae)
            running_mae += batch_mae * bs

            total += bs

            avg_loss = running_loss / total
            avg_mae = running_mae / total

            # mostramos en la barra de progreso
            train_bar.set_postfix(loss=avg_loss, mae=avg_mae)
            print(f"accuracy: {accuracy} %")


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
