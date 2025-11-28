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


