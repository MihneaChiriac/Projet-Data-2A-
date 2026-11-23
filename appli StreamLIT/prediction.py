from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DOSSIER = Path(__file__).parent
FICHIER_RATS_COMPLET = DOSSIER / "clean_rats_complet.csv"
FICHIER_METEO = DOSSIER / "clean_meteo.csv"

CIBLE = "nb_signalements"
ARRONDISSEMENTS = list(range(1, 21))
SAISONS = ["Printemps", "Ete", "Automne", "Hiver"]
SAISON_PAR_MOIS = {
    12: "Hiver", 1: "Hiver", 2: "Hiver",
    3: "Printemps", 4: "Printemps", 5: "Printemps",
    6: "Ete", 7: "Ete", 8: "Ete",
    9: "Automne", 10: "Automne", 11: "Automne",
}
# modalités de référence, sans indicatrice dans le modèle
SAISON_REFERENCE = "Printemps"
ARRONDISSEMENT_REFERENCE = 1
VARIABLES_METEO = ["MAX_TEMPERATURE_C", "pluie_veille"]
# un jour compte comme pluvieux à partir de 1 mm
SEUIL_PLUIE_MM = 1.0
# avant 2016, très peu de signalements (au plus 218 par an)
DATE_DEBUT = "2016-01-01"
# prévision hors échantillon : apprentissage avant cette date, test à partir de cette date
DATE_DEBUT_TEST = "2024-01-01"


def charger_signalements(fichier: Path) -> pd.DataFrame:
    """Lit les signalements nettoyés, avec la date en datetime et l'arrondissement en entier."""
    rats = pd.read_csv(fichier, parse_dates=["DATEDECL"])
    rats["ARRONDISSEMENT"] = rats["ARRONDISSEMENT"].astype(int)
    return rats


def charger_meteo(fichier: Path) -> pd.DataFrame:
    """Lit la météo journalière et ajoute pluie_veille (1 si la veille a reçu au moins SEUIL_PLUIE_MM).

    Renvoie DATE, MAX_TEMPERATURE_C et pluie_veille, sans le premier jour (veille inconnue).
    """
    meteo = pd.read_csv(fichier, parse_dates=["DATE"]).sort_values("DATE")
    # shift(1) ne donne la veille que s'il y a exactement un jour par ligne, sans trou
    if not meteo["DATE"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError("la météo doit contenir chaque jour exactement une fois")
    meteo["pluie_veille"] = (meteo["PRECIP_TOTAL_DAY_MM"].shift(1) >= SEUIL_PLUIE_MM).astype(int)
    return meteo.iloc[1:][["DATE", *VARIABLES_METEO]]


def construire_panel(rats: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par jour et par arrondissement, avec le nombre de signalements (0 si aucun) et la météo du jour."""
    comptes = rats.groupby(["DATEDECL", "ARRONDISSEMENT"]).size()
    comptes.index.names = ["DATE", "ARRONDISSEMENT"]
    # toutes les combinaisons jour x arrondissement, pour garder les jours sans signalement
    toutes = pd.MultiIndex.from_product([meteo["DATE"], ARRONDISSEMENTS], names=["DATE", "ARRONDISSEMENT"])
    panel = comptes.reindex(toutes, fill_value=0).rename(CIBLE).reset_index()
    # vérif : aucun signalement perdu (date hors météo ou arrondissement hors 1-20)
    if panel[CIBLE].sum() != len(rats):
        raise ValueError("des signalements ne sont pas dans le panel")
    panel = panel.merge(meteo, on="DATE", how="left", validate="many_to_one")
    panel["saison"] = panel["DATE"].dt.month.map(SAISON_PAR_MOIS)
    return panel


def filtrer_periode(panel: pd.DataFrame, debut: str, fin: str | None = None) -> pd.DataFrame:
    """Lignes du panel dont la DATE est dans [debut, fin[ ; fin=None pour aller jusqu'à la dernière date."""
    garde = panel["DATE"] >= debut
    if fin is not None:
        garde &= panel["DATE"] < fin
    return panel[garde]


def indicatrices(valeurs: pd.Series, modalites: list, prefixe: str, reference) -> pd.DataFrame:
    """Indicatrices 0/1 de toutes les modalités sauf la référence, même si certaines sont absentes de valeurs."""
    # catégories fixées : mêmes colonnes à l'estimation et à la prédiction
    categories = pd.Series(pd.Categorical(valeurs, categories=modalites), index=valeurs.index)
    return pd.get_dummies(categories, prefix=prefixe, dtype=int).drop(columns=f"{prefixe}_{reference}")


def preparer_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Matrice des variables explicatives : constante, météo, saisons et arrondissements."""
    X = pd.concat(
        [
            df[VARIABLES_METEO],
            indicatrices(df["saison"], SAISONS, "saison", SAISON_REFERENCE),
            indicatrices(df["ARRONDISSEMENT"], ARRONDISSEMENTS, "arr", ARRONDISSEMENT_REFERENCE),
        ],
        axis=1,
    )
    # has_constant="add" : sinon la constante est sautée si une colonne est constante (cas d'une prédiction)
    return sm.add_constant(X, has_constant="add")


def estimer_modele(panel: pd.DataFrame):
    """Régression linéaire (MCO) de CIBLE, écarts-types robustes à l'hétéroscédasticité (HC1)."""
    return sm.OLS(panel[CIBLE], preparer_variables(panel)).fit(cov_type="HC1")


def tableau_coefficients(resultat) -> pd.DataFrame:
    """Beta, écart-type et p-valeur de chaque variable."""
    return pd.DataFrame({"beta": resultat.params, "ecart_type": resultat.bse, "p_valeur": resultat.pvalues})


def tests_joints(resultat) -> pd.DataFrame:
    """Test de Wald (statistique F) que tous les coefficients d'un groupe de variables sont nuls."""
    noms = resultat.model.exog_names
    groupes = {
        "global (toutes sauf constante)": [nom for nom in noms if nom != "const"],
        "meteo": VARIABLES_METEO,
        "saisons": [nom for nom in noms if nom.startswith("saison_")],
        "arrondissements": [nom for nom in noms if nom.startswith("arr_")],
    }
    lignes = []
    for groupe, variables in groupes.items():
        test = resultat.f_test(", ".join(f"{variable} = 0" for variable in variables))
        lignes.append({
            "groupe": groupe,
            "nb_contraintes": len(variables),
            "F": np.asarray(test.fvalue).item(),
            "p_valeur": np.asarray(test.pvalue).item(),
        })
    return pd.DataFrame(lignes)


def evaluer_prevision(panel: pd.DataFrame, debut_test: str) -> pd.DataFrame:
    """Erreurs de prévision hors échantillon : modèle estimé avant debut_test, évalué à partir de debut_test.

    La référence prédit, pour chaque arrondissement, sa moyenne sur la période d'apprentissage :
    elle mesure ce qu'apportent la météo et la saison.
    """
    apprentissage = filtrer_periode(panel, panel["DATE"].min(), debut_test)
    test = filtrer_periode(panel, debut_test)
    modele = estimer_modele(apprentissage)
    predictions = {
        # comme pour la simulation, une prédiction négative est ramenée à 0
        "modele": np.maximum(0, modele.predict(preparer_variables(test))),
        "reference (moyenne par arrondissement)": test["ARRONDISSEMENT"].map(
            apprentissage.groupby("ARRONDISSEMENT")[CIBLE].mean()
        ),
    }
    return pd.DataFrame([
        {
            "prediction": nom,
            "R2_test": r2_score(test[CIBLE], prevu),
            "RMSE_test": np.sqrt(mean_squared_error(test[CIBLE], prevu)),
            "MAE_test": mean_absolute_error(test[CIBLE], prevu),
        }
        for nom, prevu in predictions.items()
    ])


def predire_risque_par_arrondissement(resultat, temp_max_c: float, pluie_veille: bool, saison: str) -> pd.DataFrame:
    """Nombre de signalements prédit dans chaque arrondissement pour un jour aux conditions données."""
    jour = pd.DataFrame({"ARRONDISSEMENT": ARRONDISSEMENTS})
    jour["MAX_TEMPERATURE_C"] = temp_max_c
    jour["pluie_veille"] = int(pluie_veille)
    jour["saison"] = saison
    # un modèle linéaire peut prédire un nombre négatif : ramené à 0
    jour["nb_signalements_predits"] = np.maximum(0, resultat.predict(preparer_variables(jour))).round(3)
    return jour[["ARRONDISSEMENT", "nb_signalements_predits"]].sort_values(
        "nb_signalements_predits", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    rats = charger_signalements(FICHIER_RATS_COMPLET)
    meteo = charger_meteo(FICHIER_METEO)
    panel = filtrer_periode(construire_panel(rats, meteo), DATE_DEBUT)
    resultat = estimer_modele(panel)

    print("=== RÉGRESSION LINÉAIRE : signalements par arrondissement et par jour ===")
    print(f"Observations : {int(resultat.nobs)} ({panel['DATE'].nunique()} jours x {len(ARRONDISSEMENTS)} arrondissements)")
    print(f"Période      : {panel['DATE'].min():%Y-%m-%d} -> {panel['DATE'].max():%Y-%m-%d}")
    print(f"Moyenne de {CIBLE} : {panel[CIBLE].mean():.4f}")
    print(f"R²           : {resultat.rsquared:.4f}")
    print(f"Références   : saison {SAISON_REFERENCE}, arrondissement {ARRONDISSEMENT_REFERENCE}")
    print("Écarts-types robustes à l'hétéroscédasticité (HC1)")

    print("\n--- Coefficients ---")
    print(tableau_coefficients(resultat).to_string(float_format=lambda x: f"{x:.4g}"))

    print("\n--- Tests joints (Wald, H0 : tous les coefficients du groupe sont nuls) ---")
    print(tests_joints(resultat).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n--- Simulation : jour d'été à 25 °C avec pluie la veille ---")
    print(predire_risque_par_arrondissement(resultat, temp_max_c=25.0, pluie_veille=True, saison="Ete").head().to_string(index=False))

    print(f"\n--- Prévision hors échantillon (apprentissage {DATE_DEBUT} -> {DATE_DEBUT_TEST} exclu, test ensuite) ---")
    print(f"Part de zéros sur la période de test : {(filtrer_periode(panel, DATE_DEBUT_TEST)[CIBLE] == 0).mean():.1%}")
    print(evaluer_prevision(panel, DATE_DEBUT_TEST).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
