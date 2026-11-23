import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Data/clean_rats.csv")

# PREMIERES STATISTIQUES ------------------------------------------
# Caractéristiques du tableau : 

print(data.sample(10))
print(data.dtypes)

# Pour chaque colonne où c'est pertinent, on affiche les proportions par groupe
for col in data.columns.drop(["ADRESSE", "X", "Y"]):
    print(f"--- {col} ---")
    print(data[col].value_counts(normalize=True, dropna=False).round(5)*100)
    print("____done____")

# ANALYSES --------------------------------------------------------------------------
# EVOLUTION DU NOMBRE DE SIGNALEMENTS PAR AN
axe = data['ANNEEDECL'].value_counts().sort_index().plot.bar()
axe.set_xlabel("Année")
axe.set_ylabel("Nombre de déclarations")
axe.set_title("Evolution du nombre de déclarations entre 2012 et 2026")
plt.tight_layout()
plt.savefig("Outputs/Evolution_nombre_declarations_année.png", dpi=150)
plt.show()

# HISTOGRAMME D NOMBRE DE SIGNALEMENTS PAR MOIS, TOUTES ANNEES CONFONDUES
axe = data['MOISDECL'].value_counts().sort_index().plot.bar()
axe.set_xlabel("Mois")
axe.set_ylabel("Nombre de déclarations")
plt.tight_layout()
plt.savefig("Outputs/Signalements_par_mois.png", dpi=150)
plt.show()

# HISTOGRAMME DE L'EVOLUTION DU NOMBRE DE SIGNALEMENTS SUR LA PERIODE PAR SAISON
# On rajoute la colonne saison au df
conditions = [data['MOISDECL'].isin([12, 1, 2]),
    data['MOISDECL'].isin([3, 4, 5]),
    data['MOISDECL'].isin([6, 7, 8]),
    data['MOISDECL'].isin([9, 10, 11])]
saisons =["Hiver", "Printemps", "Ete", "Automne"]
data['SAISONDECL'] = np.select(conditions, saisons, default=None)

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
plt.suptitle("Evolution du nombre de déclaration par saisons entre 2012 et 2025")
plt.subplots_adjust(top=0.90)
for ax, saison in zip(axes.flat, saisons):
    data.loc[data["SAISONDECL"]==saison, 
    "ANNEEDECL"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_xlabel("")
    ax.set_title(saison)
# même échelle sur l'axe des ordonnées
ymax = max(ax.get_ylim()[1] for ax in axes.flat)
for ax in axes.flat:
    ax.set_ylim(0, ymax)
#Améliorer la lisibilité
axes[1, 0].set_title("Ete", y=-0.25)
axes[1, 1].set_title("Automne", y=-0.25)

plt.savefig("Outputs/Evolution du nombre de déclarations par saisons entre 2012 et 2025")
plt.show()
