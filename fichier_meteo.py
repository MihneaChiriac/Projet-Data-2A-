import pandas as pd

# prise en main des données
test = pd.read_csv('Data/historique-meteo-paris-2012.csv', sep=',', comment="#")
print(test.head(3))
print(test.dtypes)
print(test.columns)

# charger toutes les données et les mettres dans un dictionnaire
#Harmonisation des colonnes
meteos = {}
for i in range(12, 26):
    print(i)
    meteos[f"meteo20{i}"] = pd.read_csv(f'Data/historique-meteo-paris-20{i}.csv', sep=',', comment="#")
    meteos[f"meteo20{i}"]["DATE"] = pd.to_datetime(meteos[f"meteo20{i}"]["DATE"])
    meteos[f"meteo20{i}"]["JOUR"] = meteos[f"meteo20{i}"]["DATE"].dt.day
    meteos[f"meteo20{i}"]["MOIS"] = meteos[f"meteo20{i}"]["DATE"].dt.month
    meteos[f"meteo20{i}"]["ANNEE"] = meteos[f"meteo20{i}"]["DATE"].dt.year
    # Température moyenne par jour
    meteos[f"meteo20{i}"]["MOYENNE_TEMPERATURE_JOUR_C"] = (meteos[f"meteo20{i}"]["TEMPERATURE_MORNING_C_6H"] +
                                                           meteos[f"meteo20{i}"]["TEMPERATURE_NOON_C_12H"] +
                                                           meteos[f"meteo20{i}"]["TEMPERATURE_EVENING_C_18H"] +
                                                           meteos[f"meteo20{i}"]["TEMPERATURE_NIGHT_C_3H"]) / 4
    # Moyenne de température par mois
    meteos[f"meteo20{i}"]["MOYENNE_MOIS"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["MOYENNE_TEMPERATURE_JOUR_C"].transform("mean").round(4)
    # Moyenne des autres variables par mois
    meteos[f"meteo20{i}"]["MOYENNE_PRECIP_DAY_MM"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["PRECIP_TOTAL_DAY_MM"].transform("mean").round(4)
    meteos[f"meteo20{i}"]["MOYENNE_MAX_HUMIDITY_PERCENT"] = meteos[f"meteo20{i}"].groupby(["MOIS"])["HUMIDITY_MAX_PERCENT"].transform("mean").round(4)

print(meteos["meteo2024"].head(3))

# Créer le dataframe propre :
liste_colonnes = ["DATE", "JOUR", "MOIS", "ANNEE", "MAX_TEMPERATURE_C", "MIN_TEMPERATURE_C",
                  "MOYENNE_TEMPERATURE_JOUR_C", "TEMPERATURE_MORNING_C_6H", "TEMPERATURE_NOON_C_12H",
                  "TEMPERATURE_EVENING_C_18H", "PRECIP_TOTAL_DAY_MM", "HUMIDITY_MAX_PERCENT",
                  "TEMPERATURE_NIGHT_C_3H", "OPINION", "MOYENNE_MOIS"]
for i in range(12,26):
    meteos[f"meteo20{i}"] = meteos[f"meteo20{i}"][liste_colonnes]

meteo_clean = pd.concat(meteos.values(), ignore_index=True)
print(meteo_clean.sample(20))
meteo_clean.to_csv('Data/clean_meteo.csv')
