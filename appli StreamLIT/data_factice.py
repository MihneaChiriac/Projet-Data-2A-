import pandas as pd
import numpy as np

def generate_mock_dataset(n_iris=50, n_days=60, output_path="mock_data.parquet"):
    """
    Génère un jeu de données factice conforme au contrat de données pour le projet, en attendant les données nettoyées.
     Contient les clés géospatiales, temporelles, la cible et toutes les features environnementales.
    """
    np.random.seed(42)
    
    # 1. Clés géographiques (IRIS Paris) et temporelles
    iris_list = [f"751{i:02d}{i%10:02d}01" for i in range(1, n_iris + 1)]
    dates = pd.date_range(end=pd.Timestamp("2025-06-30"), periods=n_days, freq="D")
    
    # Produit cartésien (IRIS x Date)
    grid = pd.MultiIndex.from_product([iris_list, dates], names=["code_iris", "date"]).to_frame().reset_index(drop=True)
    
   #localisation aléatoire mais fixe des quartiers 
    iris_coords = {
        code: (
            np.random.uniform(48.82, 48.89),
            np.random.uniform(2.28, 2.40)
        ) for code in iris_list
    }
    grid["lat"] = grid["code_iris"].map(lambda x: iris_coords[x][0])
    grid["lon"] = grid["code_iris"].map(lambda x: iris_coords[x][1])
    
    # 2. Features Temporelles & Météo
    grid["jour_semaine"] = grid["date"].dt.dayofweek
    grid["est_weekend"] = grid["jour_semaine"].isin([5, 6])
    grid["mois"] = grid["date"].dt.month
    
    # Météo simulée
    # Température maximale (°C)
    # - 3 (car on veut que l'hiver soit qu plus bas et l'été au plus grand)
    base_temp = 12 + 10 * np.sin((grid["mois"] - 3) * np.pi / 6)
    grid["temp_max_c"] = np.round(base_temp + np.random.normal(0, 3, len(grid)), 1) # "nombre de lignes len(grid)"
    
    # Précipitations (mm)
    grid["precip_mm"] = np.round(np.random.exponential(scale=2.5, size=len(grid)) * np.random.choice([0, 1], p=[0.6, 0.4], size=len(grid)), 1)
    #choice = loi de bernoulli 
    # exp car les grosses pluies sont très rares
    
    # Pluie la veille (calculé par IRIS)
    grid = grid.sort_values(["code_iris", "date"]).reset_index(drop=True)
    grid["precip_hier"] = grid.groupby("code_iris")["precip_mm"].shift(1).fillna(0)
    grid["pluie_veille"] = grid["precip_hier"] >= 2.0
    grid.drop(columns=["precip_hier"], inplace=True)
    
    # Vacances scolaires zone C (ex: Février/Avril/Juillet)
    grid["est_vacances_zone_c"] = grid["mois"].isin([2, 4, 7, 8, 12]) & (grid["date"].dt.day % 15 < 7)

    # 3. Features Spatiales & Attracteurs Environnementaux (invariables ou peu variables par IRIS)
    iris_static = {}
    for code in iris_list:
        pop = np.random.randint(1500, 6000)
        superficie_km2 = np.random.uniform(0.1, 0.5)
        iris_static[code] = {
            "nb_restos_fastfood": np.random.randint(3, 45),
            "nb_commerces_bouche": np.random.randint(5, 30),
            "nb_poubelles_containers": np.random.randint(10, 80),
            "surface_parcs_m2": np.round(np.random.choice([0, np.random.uniform(1000, 25000)], p=[0.4, 0.6]), 1),
            "dist_eau_m": np.round(np.random.uniform(30, 2000), 1),
            "pop_totale_iris": pop,
            "densite_pop_km2": np.round(pop / superficie_km2, 1)
        }
    
    static_df = pd.DataFrame.from_dict(iris_static, orient="index").reset_index().rename(columns={"index": "code_iris"})
    grid = pd.merge(grid, static_df, on="code_iris")

    # Features dynamiques locales (Marchés, Chantiers)
    grid["presence_marche_jour"] = np.random.choice([True, False], size=len(grid), p=[0.15, 0.85])
    grid["nb_chantiers_actifs"] = np.random.choice([0, 1, 2, 3], size=len(grid), p=[0.5, 0.3, 0.15, 0.05])

   # 4. Génération de la Cible (Régression linéaire sous-jacente)
    y_pred = (
        5.0                                              
        + 0.25 * (grid["nb_restos_fastfood"] - 15)       
        + 0.15 * (grid["temp_max_c"] - 15)               
        + 2.0 * grid["pluie_veille"].astype(int)         
        + 1.5 * grid["presence_marche_jour"].astype(int)   
        + 1.0 * grid["nb_chantiers_actifs"]              
        - 0.002 * grid["dist_eau_m"]                    
        + 1.0 * (grid["pop_totale_iris"] / 3000)         
    )

    # Ajout du bruit  N(0, sigma^2) propre à la régression linéaire
    bruit = np.random.normal(loc=0.0, scale=2.0, size=len(grid))
    y_bruite = y_pred + bruit

    # Arrondi à l'entier et seuillage à 0 (un nombre de signalements ne peut pas être négatif)
    grid["nb_signalements"] = np.maximum(0, np.round(y_bruite)).astype(int)
    
    # Normalisation du taux pour 10 000 habitants
    grid["taux_signalements_10k_hab"] = np.round((grid["nb_signalements"] / grid["pop_totale_iris"]) * 10000, 2)

    # Réorganisation propre des colonnes selon le contrat de données
    cols_order = [
        "code_iris", "date", "lat", "lon",
        "nb_signalements", "taux_signalements_10k_hab",
        "temp_max_c", "precip_mm", "pluie_veille", "jour_semaine", "est_weekend", "mois", "est_vacances_zone_c",
        "nb_restos_fastfood", "nb_commerces_bouche", "nb_poubelles_containers", "surface_parcs_m2", "dist_eau_m",
        "presence_marche_jour", "nb_chantiers_actifs",
        "pop_totale_iris", "densite_pop_km2"
    ]
    grid = grid[cols_order].sort_values(["code_iris", "date"]).reset_index(drop=True)

    # Exportation Parquet
    grid.to_parquet(output_path, index=False)
    print(f"Dataset factice généré avec succès ({len(grid)} lignes, {len(cols_order)} colonnes) -> {output_path}")
    return grid

if __name__ == "__main__":
    generate_mock_dataset(n_iris=50, n_days=60, output_path="/workspace/scratch/mock_data.parquet")
