import pandas as pd

data2012 = pd.read_csv('Data/DMR_2012.csv', sep=";", encoding='utf-8')
data2013 = pd.read_csv('Data/DMR_2013.csv', sep=";", encoding='utf-8')
data2014 = pd.read_csv('Data/DMR_2014.csv', sep=";", encoding='utf-8')
data2015 = pd.read_csv('Data/DMR_2015.csv', sep=";", encoding='utf-8')
data2016 = pd.read_csv('Data/DMR_2016.csv', sep=";", encoding='utf-8')
data2017 = pd.read_csv('Data/DMR_2017.csv', sep=";", encoding='utf-8')
data2018 = pd.read_csv('Data/DMR_2018.csv', sep=";", encoding='utf-8')
data2019 = pd.read_csv('Data/DMR_2019.csv', sep=";", encoding='utf-8')
data2020 = pd.read_csv('Data/DMR_2020.csv', sep=";", encoding='utf-8')
data2021 = pd.read_csv('Data/DMR_2021.csv', sep=";", encoding='utf-8')
data2022 = pd.read_csv('Data/DMR_2022.csv', sep=";", encoding='cp1252')
data2023 = pd.read_csv('Data/DMR_2023.csv', sep=";", encoding='cp1252')
data2024 = pd.read_csv('Data/DMR_2024.csv', sep=";", encoding='utf-8')
data2025 = pd.read_csv('Data/DMR_2025.csv', sep=";", encoding='utf-8')

####### HARMONISATION DES COLONNES

datas = {f"data20{i}": globals()[f"data20{i}"] for i in range(12, 26)}

liste_colonnes = ["ID_DMR", "TYPE", "SOUSTYPE", "ADRESSE", "CODE_POSTAL", "VILLE", "ARRONDISSEMENT",
 "ANNEEDECL", "MOISDECL", "CONSEILQUARTIER", "X", "Y"]

for i in range(12,26):
    # on met tous les noms de colonnes en majuscule
    datas[f"data20{i}"].columns = datas[f"data20{i}"].columns.str.upper()
    datas[f"data20{i}"] = datas[f"data20{i}"][liste_colonnes]


######## IDENTIFICATION DES LIGNES SIGNALANT UNE PRESENCE DE RATS

sous_types_pertinents = {}
for i in range(12, 26):
    df = datas[f"data20{i}"]
    sous_types = df["SOUSTYPE"].dropna().unique()
    sous_types_pertinents[f"20{i}"] = [s for s in sous_types if isinstance(s, str) and (" rat" in s.lower() or " rats" in s.lower() or " souris" in s.lower()) and ("ratp" not in s.lower())]

sous_types_pertinents_uniques = []
for s in sous_types_pertinents.values():
    for u in s:
        if u not in sous_types_pertinents_uniques:
            sous_types_pertinents_uniques.append(u)

''' on trouve : [' Présence de rats', ' Présence de souris', " Autres problèmes d'animaux et d'insectes : [] Rats sortant des égouts"]'''

# SELECTION DES LIGNES PERTINENTES SELON LE SOUS TYPE

for i in range(12, 26):
    df = datas[f"data20{i}"]
    datas[f"data20{i}"] = df[df["SOUSTYPE"].isin(sous_types_pertinents_uniques)]
    print(datas[f"data20{i}"].head(2))

# CONSTRUCTION DU FICHIER PROPRE
data_clean = pd.concat([datas[f"data20{i}"] for i in range(12,26)])
print(data_clean.sample(n=20))

data_clean.to_csv('Data/clean_rats.csv')