import os
import pandas as pd
import numpy as np

def run_eda(file_path="mock_data.parquet"):

    # 1. Chargement des données
    df = pd.read_parquet(file_path)
    print(f" Dimensions du jeu de données : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")

    # 2. Aperçu des colonnes et types
    print("--- 1. STRUCTURE ET TYPES DES DONNÉES ---")
    info_df = pd.DataFrame({
        "Type": df.dtypes,
        "Valeurs Manquantes": df.isnull().sum(),
        "% Manquant": (df.isnull().sum() / len(df) * 100).round(2),
        "Valeurs Uniques": df.nunique()
    })
    print(info_df.to_string())

    # 3. Distribution de la variable cible
    print("\n--- 2. ANALYSE DE LA VARIABLE CIBLE (nb_signalements) ---")
    target = df["nb_signalements"]
    mean_val = target.mean()
    var_val = target.var()
    
    print(f"Moyenne           : {mean_val:.2f}")
    print(f"Médiane           : {target.median():.2f}")
    print(f"Écart-type        : {target.std():.2f}")
    print(f"Variance          : {var_val:.2f}")
    print(f"Ratio Var/Moyenne : {var_val / mean_val:.2f} (Indice de surdispersion)")
    print(f"Minimum           : {target.min()}")
    print(f"Maximum           : {target.max()}")
    print(f"Total signalements: {target.sum()}")

    # 4. Analyse par Quartier (IRIS)
    print("\n--- 3. DISTRIBUTION DES SIGNALEMENTS PAR QUARTIER (IRIS) ---")
    stats_iris = df.groupby("code_iris")["nb_signalements"].agg(
        Total_Signalements="sum",
        Moyenne_Journaliere="mean",
        Max_Journalier="max",
        Std_Journalier="std"
    ).reset_index()

    print(f"Nombre total de quartiers IRIS analysés : {len(stats_iris)}")
    print("\n Top 5 des quartiers les plus touchés :")
    print(stats_iris.sort_values(by="Total_Signalements", ascending=False).head().to_string(index=False))

    print("\n Top 5 des quartiers les moins touchés :")
    print(stats_iris.sort_values(by="Total_Signalements", ascending=True).head().to_string(index=False))

    # 5. Corrélations avec la variable cible
    print("\n--- 4. CORRÉLATION DES FEATURES AVEC LA CIBLE ---")
    num_cols = df.select_dtypes(include=[np.number, bool]).columns
    corr_matrix = df[num_cols].corr()
    
    if "nb_signalements" in corr_matrix.columns:
        target_corr = corr_matrix["nb_signalements"].drop("nb_signalements").sort_values(ascending=False)
        corr_df = pd.DataFrame({
            "Feature": target_corr.index,
            "Coefficient de Corrélations (Pearson)": target_corr.values.round(3)
        })
        print(corr_df.to_string(index=False))

    # 6. Impact des variables catégorielles / booléennes
    print("\n--- 5. IMPACT DES FACTEURS ÉVÉNEMENTIELS ---")
    bool_cols = ["pluie_veille", "presence_marche_jour", "est_weekend", "est_vacances_zone_c"]
    for col in bool_cols:
        if col in df.columns:
            group = df.groupby(col)["nb_signalements"].agg(["mean", "std", "count"])
            group.columns = ["Moyenne_Signalements", "Ecart_type", "Nombre_Jours_Quartiers"]
            print(f"\nEffet de '{col}':")
            print(group.to_string())


if __name__ == "__main__":
    import sys
    sys.path.append("/workspace/artifacts")
    run_eda()
