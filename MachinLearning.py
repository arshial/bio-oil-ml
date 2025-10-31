import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Caricamento dei file Excel
train_df = pd.read_excel("Train.xlsx")
test_df = pd.read_excel("Test.xlsx")

# Separazione delle feature (X) e target (y)
X = train_df.drop(columns=["yield(wt%)"], errors="ignore")
y = train_df["yield(wt%)"]
X_test = test_df.copy()

# Imputazione dei valori mancanti con la media
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# Standardizzazione dei dati
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Divisione del dataset in train e validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# Inizializzazione e training del modello XGBRegressor con i migliori parametri trovati
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=6000,
    max_depth=3,
    learning_rate=0.1,
    random_state=1
)
model.fit(X_train, y_train)

# Predizione e valutazione sul validation set
y_val_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Validation MAE: {mae:.2f}")
print(f"Validation R2 Score: {r2:.2f}")

# Predizione sul set di test
test_predictions = model.predict(X_test)

# Creazione del file di submission
submission = pd.DataFrame({
    "ID": range(len(test_predictions)),
    "yield": test_predictions
})
submission.to_excel("submission_XGBRegressor.xlsx", index=False)
print("✅ submission_XGBRegressor.xlsx creato con successo")