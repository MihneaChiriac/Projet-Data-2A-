import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


# A changer pour le vrai modèle
#chaque individu est un quartier à une date précise

FEATURE_COLS = [
    "nb_restos_fastfood",
    "temp_max_c",
    "pluie_veille",
    "presence_marche_jour",
    "nb_chantiers_actifs",
    "dist_eau_m",
    "pop_totale_iris"
]

TARGET_COL = "nb_signalements"

def load_data(file_path="mock_data.parquet"):
    df = pd.read_parquet(file_path)
    return df

def train_linear_model(df):
    # Préparation des variables
    X = df[FEATURE_COLS].copy()
    
    # Conversion des booléens en int 
    if "pluie_veille" in X.columns and X["pluie_veille"].dtype == bool:
        X["pluie_veille"] = X["pluie_veille"].astype(int)
    if "presence_marche_jour" in X.columns and X["presence_marche_jour"].dtype == bool:
        X["presence_marche_jour"] = X["presence_marche_jour"].astype(int)
        
    y = df[TARGET_COL]

    # Entraînement de la régression linéaire
    model = LinearRegression()
    model.fit(X, y)
    
    # Prédictions et calcul des métriques
    y_pred = model.predict(X)
    # Le nombre de signalements prédits ne peut pas être négatif
    y_pred = np.maximum(0, y_pred)
    
    metrics = {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred)
    }
    
    # Extraction des coefficients
    coef_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)
    
    return model, metrics, coef_df, model.intercept_

def predict_risk_by_quartier(model, df_iris_features, temp_max_c, pluie_veille):
    """
    Prédit le nombre de signalements pour CHAQUE quartier IRIS
    selon des conditions météo données.
    
    Args:
        model: Modèle LinearRegression entraîné.
        df_iris_features: DataFrame contenant les caractéristiques fixes de chaque IRIS 
                          (ex: nb_restos_fastfood, dist_eau_m, pop_totale_iris).
        temp_max_c (float): Température maximale du jour à simuler.
        pluie_veille (bool/int): 1 s'il a plu la veille, 0 sinon.
        
    Returns:
        pd.DataFrame: Table par quartier avec la prédiction du nombre de signalements.
    """
    # On prend une ligne unique par IRIS (features fixes)
    iris_df = df_iris_features.drop_duplicates(subset=["code_iris"]).copy()
    
    # On injecte les conditions météo simulées
    iris_df["temp_max_c"] = temp_max_c
    iris_df["pluie_veille"] = int(pluie_veille)
    
    if "presence_marche_jour" not in iris_df.columns:
        iris_df["presence_marche_jour"] = 0
    if "nb_chantiers_actifs" not in iris_df.columns:
        iris_df["nb_chantiers_actifs"] = 0
        
    X_input = iris_df[FEATURE_COLS].copy()
    
    # Prédiction
    preds = model.predict(X_input)
    iris_df["nb_signalements_predits"] = np.round(np.maximum(0, preds), 1)
    
    # Calcul du taux de risque pour 10k hab
    iris_df["taux_risque_10k_hab"] = np.round(
        (iris_df["nb_signalements_predits"] / iris_df["pop_totale_iris"]) * 10000, 2
    )
    
    result_cols = ["code_iris", "lat", "lon", "nb_signalements_predits", "taux_risque_10k_hab", "pop_totale_iris"]
    existing_cols = [c for c in result_cols if c in iris_df.columns]
    
    return iris_df[existing_cols].sort_values(by="nb_signalements_predits", ascending=False)

if __name__ == "__main__":
    print("=== TEST DU MODULE DE PRÉDICTION LINÉAIRE ===")
    
    # 1. Chargement des données
    df_data = load_data("mock_data.parquet")
    print(f"Données chargées : {len(df_data)} lignes, {df_data['code_iris'].nunique()} quartiers IRIS.")
    
    # 2. Entraînement du modèle
    model, metrics, coef_df, intercept = train_linear_model(df_data)
    
    print("\n--- Performances du modèle ---")
    print(f"Intercept (Constante) : {intercept:.3f}")
    print(f"R² Score : {metrics['r2']:.4f}")
    print(f"RMSE     : {metrics['rmse']:.3f}")
    print(f"MAE      : {metrics['mae']:.3f}")
    
    print("\n--- Coefficients appris (Poids des facteurs) ---")
    print(coef_df.to_string(index=False))
    
    # 3. Test de prédiction par quartier pour une journée chaude après de la pluie
    print("\n--- Simulation : Journée à 25°C avec pluie la veille ---")
    predictions_quartiers = predict_risk_by_quartier(
        model=model,
        df_iris_features=df_data,
        temp_max_c=25.0,
        pluie_veille=True
    )
    
    print("\nTop 5 des quartiers les plus à risque prédits :")
    print(predictions_quartiers.head(5).to_string(index=False))
