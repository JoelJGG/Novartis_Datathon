import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "SUBMISSION" / "DataFiles" / "TRAIN"

# Cargar datasets
df_generics_train = pd.read_csv(DATA_DIR / "df_generics_train.csv")
df_med_train = pd.read_csv(DATA_DIR / "df_medicine_info_train.csv")
df_volume_train = pd.read_csv(DATA_DIR / "df_volume_train.csv")

# Merge base: volumen + info de medicamentos
df_base = pd.merge(
    df_volume_train,
    df_med_train,
    on=["country", "brand_name"],
    how="left"
)

# Añadir n_gxs desde su fuente autoritativa (df_generics_train)
df_all_merge = pd.merge(
    df_base,
    df_generics_train[["country", "brand_name", "months_postgx", "n_gxs"]],
    on=["country", "brand_name", "months_postgx"],
    how="left"
)

# Asegurar tipo numérico en months_postgx
df_all_merge["months_postgx"] = pd.to_numeric(df_all_merge["months_postgx"], errors="coerce")

# Aplicar regla: n_gxs = 0 en negativos
df_all_merge.loc[df_all_merge["months_postgx"] < 0, "n_gxs"] = 0

# Convertir columnas biológicas y pequeñas moléculas a float
cols = ["biological", "small_molecule"]
df_all_merge[cols] = df_all_merge[cols].astype(float)

# Imputar valores faltantes
imputer = KNNImputer(n_neighbors=10)
df_all_merge[["n_gxs"]] = imputer.fit_transform(df_all_merge[["n_gxs"]])
df_all_merge[["hospital_rate"]] = imputer.fit_transform(df_all_merge[["hospital_rate"]])

# One hot encoding de variables categóricas
df_all_merge = pd.get_dummies(
    df_all_merge,
    columns=["country","brand_name","month", "ther_area", "main_package"],
    drop_first=False,
    dtype=float
)

# Mover la columna 'volume' al final
cols = [c for c in df_all_merge.columns if c != "volume"] + ["volume"]
df_all_merge = df_all_merge[cols]

# Evaluar importancia de características usando Random Forest
X = df_all_merge.iloc[:, :-1].copy()
y = df_all_merge.iloc[:, -1].copy()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_

# Crear DataFrame con nombres de columnas y su importancia
feat_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
})

# Ordenar de mayor a menor
feat_importances = feat_importances.sort_values(by='importance', ascending=False)

# Seleccionar las 20 mejores
top_20 = feat_importances.head(20)

pd.set_option("display.max_rows", 100)
print(top_20)

# Verificar columnas y primeras filas
print(df_all_merge.head())
