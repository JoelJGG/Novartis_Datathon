import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer

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

# Obtener todas las columnas
cols = list(df_all_merge.columns)

# Reordenar: dejar todo igual pero mover n_gxs y volume
cols.remove("n_gxs")
cols.remove("volume")

# Insertar n_gxs justo después de months_postgx
idx = cols.index("months_postgx") + 1
cols.insert(idx, "n_gxs")

# Añadir volume al final
cols.append("volume")

# Reordenar DataFrame
df_all_merge = df_all_merge[cols]

cols = ["biological", "small_molecule"]
df_all_merge[cols] = df_all_merge[cols].astype(float)

imputer = KNNImputer(n_neighbors=10)
df_all_merge[["n_gxs"]] = imputer.fit_transform(df_all_merge[["n_gxs"]])
df_all_merge[["hospital_rate"]] = imputer.fit_transform(df_all_merge[["hospital_rate"]])

### Hasta aqui funciona bien el merge y limpieza básica ###
# One hot encoding de variables categóricas


# Verificar columnas y primeras filas
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
print(df_all_merge.head(300))
