from pathlib import Path

import pandas as pd

from eau import ajouter_distance_eau, charger_cours_d_eau

DOSSIER_DONNEES = Path(__file__).parent / "données"
FICHIER_RATS_ENRICHI = DOSSIER_DONNEES / "clean_rats_enrichi.csv"
FICHIER_RATS_COMPLET = DOSSIER_DONNEES / "clean_rats_complet.csv"

# colonnes constantes, en doublon ou sans signification connue
COLONNES_A_SUPPRIMER = [
    "ID_DMR",
    "CONSEILQUARTIER",
    "TYPE",
    "SOUSTYPE",
    "ADRESSE",
    "CODE_POSTAL",
    "ANNEEDECL",
    "MOISDECL",
    "JOURDECL",
    "dans_bois",
    "OPINION",
    "MOYENNE_MOIS",
    # températures : on ne garde que MAX_TEMPERATURE_C et MIN_TEMPERATURE_C
    "MOYENNE_TEMPERATURE_JOUR_C",
    "TEMPERATURE_MORNING_C_6H",
    "TEMPERATURE_NOON_C_12H",
    "TEMPERATURE_EVENING_C_18H",
    "TEMPERATURE_NIGHT_C_3H",
]


def enrichir_avec_eau(fichier_entree: Path, fichier_sortie: Path) -> pd.DataFrame:
    # index_col=0 : la première colonne sans nom est l'ancien index
    rats = pd.read_csv(fichier_entree, index_col=0)
    eau = charger_cours_d_eau()
    rats = ajouter_distance_eau(rats, eau)
    # erreur si une colonne de la liste n'existe pas, plutôt qu'un oubli silencieux
    rats = rats.drop(columns=COLONNES_A_SUPPRIMER)
    # sans ID_DMR, plusieurs signalements du même lieu le même jour deviennent identiques :
    # on les garde tous, chacun reste un signalement distinct
    # format imposé : une date mal écrite fait planter au lieu d'être devinée
    rats["DATEDECL"] = pd.to_datetime(rats["DATEDECL"], format="%Y-%m-%d")
    # index=False : l'ancien index, non unique, n'est pas réécrit
    rats.to_csv(fichier_sortie, index=False)
    return rats


if __name__ == "__main__":
    rats = enrichir_avec_eau(FICHIER_RATS_ENRICHI, FICHIER_RATS_COMPLET)
    print(f"{len(rats)} signalements, {rats.shape[1]} colonnes -> {FICHIER_RATS_COMPLET}")
