import pandas as pd

# prise en main des données
test = pd.read_csv('Data/historique-meteo-paris-2012.csv', sep=',', comment="#")
print(test.head(3))
print(test.dtypes)
print(test.columns)
# ON REGARDE LES COLONNES SOUHAITEES:
liste_colonnes_pertinentes = ["DATE", "MAX_TEMPERATURE_C", "MIN_TEMPERATURE_C",
                              "TEMPERATURE_MORNING_C_6H", "TEMPERATURE_NOON_C_12H",
                              "TEMPERATURE_EVENING_C_18H", "PRECIP_TOTAL_DAY_MM",
                              "HUMIDITY_MAX_PERCENT", "TEMPERATURE_NIGHT_C_3H", "OPINION"]
# charger toutes les données et les mettres dans un dictionnaire
meteos = {}
for i in range(12, 26):
    print(i)
    meteos[f"meteo20{i}"] = pd.read_csv(f'Data/historique-meteo-paris-20{i}.csv', sep=',', comment="#")
    meteos[f"meteo20{i}"] = meteos[f"meteo20{i}"][liste_colonnes_pertinentes]
    meteos[f"meteo20{i}"]["DATE"] = pd.to_datetime(meteos[f"meteo20{i}"]["DATE"])
    meteos[f"meteo20{i}"]["MOIS"] = meteos[f"meteo20{i}"]["DATE"].dt.month
    meteos[f"meteo20{i}"]["ANNEE"] = meteos[f"meteo20{i}"]["DATE"].dt.year
    # Température moyenne par jour
    meteos[f"meteo20{i}"]["MOYENNE_TEMPERATURE_JOUR_C"] = (meteos[f"meteo20{i}"]["TEMPERATURE_MORNING_C_6H"] +
                                                           meteos[f"meteo20{i}"]["TEMPERATURE_NOON_C_12H"] +
                                                           meteos[f"meteo20{i}"]["TEMPERATURE_EVENING_C_18H"] +
                                                           meteos[f"meteo20{i}"]["TEMPERATURE_NIGHT_C_3H"]) / 4
    # Moyenne de température par mois
    meteos[f"meteo20{i}"]["MOYENNE_MOIS"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["MOYENNE_TEMPERATURE_JOUR_C"].transform("mean").round(4)
    meteos[f"meteo20{i}"]["MOYENNE_MIN_MOIS"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["MIN_TEMPERATURE_C"].transform("mean").round(4)
    meteos[f"meteo20{i}"]["MOYENNE_MAX_MOIS"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["MAX_TEMPERATURE_C"].transform("mean").round(4)
    # Moyenne des autres variables par mois
    meteos[f"meteo20{i}"]["MOYENNE_PRECIP_DAY_MM"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["PRECIP_TOTAL_DAY_MM"].transform("mean").round(4)
    meteos[f"meteo20{i}"]["MOYENNE_MAX_HUMIDITY_PERCENT"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["HUMIDITY_MAX_PERCENT"].transform("mean").round(4)

print(meteos["meteo2024"].head(3))

# Créer le dataframe propre :
resultats = {}
for elt, df in meteos.items():
    resultats[elt] = (
        df.groupby(["ANNEE", "MOIS"])[["MOYENNE_MOIS", "MOYENNE_MIN_MOIS", "MOYENNE_MAX_MOIS",
                                       "MOYENNE_PRECIP_DAY_MM", "MOYENNE_MAX_HUMIDITY_PERCENT"]].first().reset_index()
                                       )
meteo_clean = pd.concat(resultats.values(), ignore_index=True)
print(meteo_clean.sample(20))
meteo_clean.to_csv('Data/meteo_clean.csv')
