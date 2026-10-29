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

#retouches finales:
print("____________DEBUT MODIF LOUIS____________")
#data_clean = data_clean.drop(columns=["Unnamed: 0"]) #on supprime une colonne inutile et sans info
print(data_clean.dtypes)
print(list(data_clean.columns.values))
print(data_clean.shape)

#pour toute colonne, afficher la proptable
for col in data_clean.columns.drop(["ADRESSE", "X", "Y"]):
    print(f"--- {col} ---")
    print(data_clean[col].value_counts(normalize=True, dropna=False).round(5)*100)
    print(f"____done: {col}____")

#test présence NaN
na_counts = data_clean.isna().sum()
print(na_counts[na_counts > 0].sort_values(ascending=False))
#résultat: 1 seul NaN, dans CONSEILQUARTIER

#conversion en nombres

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


data_clean["ARRONDISSEMENT"] = to_num(data_clean["ARRONDISSEMENT"])
data_clean["CODE_POSTAL"] = to_num(data_clean["CODE_POSTAL"])
data_clean["X"] = to_num(data_clean["X"]) #X: longitude
data_clean["Y"] = to_num(data_clean["Y"]) #Y: latitude

#vérification opération conversion:
#print(data_clean["ARRONDISSEMENT"].value_counts(normalize=True, dropna=False).round(5)*100) #vérification valeurs
#print(data_clean["CODE_POSTAL"].value_counts(normalize=True, dropna=False).round(5)*100) #vérification valeurs
#na_counts = data_clean.isna().sum()
#print(na_counts[na_counts > 0].sort_values(ascending=False)) #vérification NaN, aucun n'est apparu
#print(data_clean.dtypes) #vérification types





# on regarde le ID_DMR qui apparaît 2 fois, (constaté dans les proptables précédentes)
# print(data_clean.loc[data_clean["ID_DMR"]=="S2017D000601",:])
# c'est bien la même ligne, on vire le dupliqué
data_clean = data_clean.drop_duplicates(subset="ID_DMR")
# print(data_clean.shape) #suppression a fonctionné


# réunion des valeurs similaires de SOUSTYPE:
# print(data_clean["SOUSTYPE"].value_counts(normalize=True, dropna=False).round(5)*100) #rappel distribution
# on réecrit les catégories

data_clean.loc[data_clean["SOUSTYPE"] == " Présence de rats", "SOUSTYPE"] = "Présence de rats"
data_clean.loc[data_clean["SOUSTYPE"] == " Présence de souris", "SOUSTYPE"] = "Présence de souris"
data_clean.loc[data_clean["SOUSTYPE"].str.contains("égouts", na=False), "SOUSTYPE"] = "Rats sortant des égouts"

# affichage proptable, a-ce fonctionné? oui!
print(data_clean["SOUSTYPE"].value_counts(normalize=True, dropna=False).round(5)*100)


# l'info de VILLE est redondante à celle de ARRONDISSEMENT, on supprime VILLE
data_clean = data_clean.drop(columns=["VILLE"])

# final check:
print(data_clean.dtypes)
print(data_clean.shape)
for col in data_clean.columns.drop(["ADRESSE", "X", "Y"]):
    print(f"--- CHECK FINAL: {col} ---")
    print(data_clean[col].value_counts(normalize=True, dropna=False).round(5)*100)
    print(f"____FIN CHECK FINAL: {col}____")
# tout paraît ok

data_clean.to_csv('Data/clean_rats.csv')