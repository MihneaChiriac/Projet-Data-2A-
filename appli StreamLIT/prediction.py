import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


# A changer pour le vrai modèle
#chaque individu est un signalement à une date précise
#j'ai pas mis les quartiers car cela rajouterait de la colinéarité dans le modèle

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

def derive_saison(df):
    df_out = df.copy()
    if "saison" not in df_out.columns:
        if "mois" in df_out.columns:
            mois_series = df_out["mois"]
        elif "date" in df_out.columns:
            mois_series = pd.to_datetime(df_out["date"]).dt.month
        else:
            mois_series = pd.Series(6, index=df_out.index) # Par défaut été
            
        def mois_to_saison(m):
            if m in [12, 1, 2]:
                return "Hiver"
            elif m in [3, 4, 5]:
                return "Printemps"
            elif m in [6, 7, 8]:
                return "Ete"
            else:
                return "Automne"
                
        df_out["saison"] = mois_series.apply(mois_to_saison)
    return df_out

def prepare_features(df):
    """
    Prépare le DataFrame avec l'encodage One-Hot des saisons (base = Printemps) (on le met donc pas).
    """
    df_prep = derive_saison(df)
    
    # Encodage One-Hot pour la saison
    saison_dummies = pd.get_dummies(df_prep["saison"], prefix="saison", dtype=int)
    
    # S'assurer que toutes les colonnes de saison sont présentes
    for s in ["saison_Automne", "saison_Ete", "saison_Hiver", "saison_Printemps"]:
        if s not in saison_dummies.columns:
            saison_dummies[s] = 0
            
    # Les colonnes de saisons ajoutées :
    saison_cols = ["saison_Ete", "saison_Automne", "saison_Hiver"]
    
    X = pd.concat([df_prep[FEATURE_COLS], saison_dummies[saison_cols]], axis=1)
    
    # Convertir les booléens
    if "pluie_veille" in X.columns and X["pluie_veille"].dtype == bool:
        X["pluie_veille"] = X["pluie_veille"].astype(int)
    if "presence_marche_jour" in X.columns and X["presence_marche_jour"].dtype == bool:
        X["presence_marche_jour"] = X["presence_marche_jour"].astype(int)
        
    return X, saison_cols

def load_data(file_path="mock_data_linear.parquet"):
    return pd.read_parquet(file_path)

def train_linear_model(df):
 
    X, saison_cols = prepare_features(df)
    y = df[TARGET_COL]

    feature_names = list(X.columns)

    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = np.maximum(0, model.predict(X))
    
    metrics = {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred)
    }
    
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)
    
    return model, metrics, coef_df, model.intercept_, feature_names

def predict_risk_by_quartier(model, df_iris_features, temp_max_c, pluie_veille, saison="Ete"):
    iris_df = df_iris_features.drop_duplicates(subset=["code_iris"]).copy()
    
    iris_df["temp_max_c"] = temp_max_c
    iris_df["pluie_veille"] = int(pluie_veille)
    iris_df["saison"] = saison
    
    if "presence_marche_jour" not in iris_df.columns:
        iris_df["presence_marche_jour"] = 0
    if "nb_chantiers_actifs" not in iris_df.columns:
        iris_df["nb_chantiers_actifs"] = 0
        
    X_input, _ = prepare_features(iris_df)
    
    preds = model.predict(X_input)
    
    iris_df["nb_signalements_predits"] = np.round(np.maximum(0, preds), 1)
    iris_df["taux_risque_10k_hab"] = np.round(
        (iris_df["nb_signalements_predits"] / iris_df["pop_totale_iris"]) * 10000, 2
    )  # nb de signalements pour 10000 hab 
    
    return iris_df[["code_iris", "lat", "lon", "nb_signalements_predits", "taux_risque_10k_hab"]].sort_values(
        by="nb_signalements_predits", ascending=False
    )

if __name__ == "__main__":
    print("=== TEST DU MODULE DE PRÉDICTION LINÉAIRE AVEC SAISONS ===")
    df_data = load_data("mock_data.parquet")
    print(f"Données chargées : {len(df_data)} lignes, {df_data['code_iris'].nunique()} quartiers IRIS.")
    
    model, metrics, coef_df, intercept, feature_names = train_linear_model(df_data)
    
    print("\n--- Performances du modèle ---")
    print(f"Intercept (Constante) : {intercept:.3f}")
    print(f"R² Score : {metrics['r2']:.4f}")
    print(f"RMSE     : {metrics['rmse']:.3f}")
    print(f"MAE      : {metrics['mae']:.3f}")
    
    print("\n--- Coefficients appris (Poids des facteurs & Saisons) ---")
    print(coef_df.to_string(index=False))
    
    print("\n--- Simulation : Journée d'Été à 25°C avec pluie la veille ---")
    sim_res = predict_risk_by_quartier(model, df_data, temp_max_c=25.0, pluie_veille=True, saison="Ete")
    print("\nTop 5 des quartiers les plus à risque prédits :")
    print(sim_res.head().to_string(index=False))
