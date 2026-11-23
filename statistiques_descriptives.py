import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("Data/clean_rats.csv")

# PREMIERES STATISTIQUES ------------------------------------------
# Caractéristiques du tableau :

print(data.sample(10))
print(data.dtypes)

# Pour chaque colonne où c'est pertinent, on affiche les proportions par groupe
for col in data.columns.drop(["ID_DMR", "DATEDECL", "CODE_POSTAL", "TYPE", "JOURDECL", "ADRESSE", "X", "Y"]):
    print(f"--- {col} ---")
    print(data[col].value_counts(normalize=True, dropna=False).round(5)*100)
    print("____done____")

# On veut regarder les mois, années, arrondissements et quartier les plus
# et moins touchés pour davantage de lisibilité. Sous forme de tableau.
colonnes = ['MOISDECL', "ANNEEDECL", 'ARRONDISSEMENT', 'CONSEILQUARTIER']
resultats = []
for col in colonnes:
    groupes = data.groupby(col, dropna=False).size()
    plus = groupes.idxmax()
    moins = groupes.idxmin()
    resultats.append([col, str(plus), int(groupes.max()),
                      str(moins), int(groupes.min())])
maximin = pd.DataFrame(
    resultats,
    columns=['Variable', 'Plus touché', 'Effectif max', 'Moins touché', 'Effectif min'])

fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')
ax.set_title("Tableau descriptif des lieux et périodes où il y a le plus de signalements",
             fontweight='bold', pad=15)
tab = ax.table(
    cellText=maximin.values,
    colLabels=maximin.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.30, 0.30, 0.30, 0.30, 0.30]
)
tab.auto_set_font_size(False)
tab.set_fontsize(11)
tab.scale(1.2, 1.8)

plt.savefig('Outputs/tableau_extrema.png',
            dpi=300,
            bbox_inches='tight',
            transparent=False)

plt.close()

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
# calculons les moyennes mensuelles
comptages = data.groupby(['ANNEEDECL', 'MOISDECL']).size()
moyennes = comptages.groupby('MOISDECL').mean()
axe = data['MOISDECL'].value_counts(normalize=False).sort_index().plot.bar()
for i, moyenne in enumerate(moyennes):
    axe.hlines(
        y=moyenne,
        xmin=i-0.4,
        xmax=i+0.4,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Moyenne mensuelle' if i == 0 else None
    )
axe.set_title('Moyenne du nombre de signalements par mois')
axe.set_xlabel("Mois")
axe.set_ylabel("Moyenne des signalements mensuels")
handles, labels = axe.get_legend_handles_labels()
print(axe.get_legend_handles_labels()) 
print(len(handles))
print(labels)  # pour pouvoir personnaliser la légende
axe.legend(
    handles=[handles[1], handles[0]],
    labels=['Nombre de signalements', 'Moyenne'],
    loc='best'
)
plt.tight_layout()
plt.savefig("Outputs/Signalements_par_mois.png", dpi=150)
plt.show()
print(f"mois le plus infesté : {moyennes.idxmax()}, en moyenne {moyennes.max()} signalements.")
print(f"mois le moins infesté : {moyennes.idxmin()}, en moyenne {moyennes.min()} signalements")

# HISTOGRAMME DE L'EVOLUTION DU NOMBRE DE SIGNALEMENTS SUR LA PERIODE PAR SAISON
# On rajoute la colonne saison au df
conditions = [data['MOISDECL'].isin([12, 1, 2]),
              data['MOISDECL'].isin([3, 4, 5]),
              data['MOISDECL'].isin([6, 7, 8]),
              data['MOISDECL'].isin([9, 10, 11])]
saisons = ["Hiver", "Printemps", "Ete", "Automne"]
data['SAISONDECL'] = np.select(conditions, saisons, default=None)

# pour comparer avec la tendance générale :
annees = sorted(data["ANNEEDECL"].dropna().unique())
tendance_annuelles = (
    data.groupby("ANNEEDECL")
    .size()
    .reindex(annees, fill_value=0)
)

# Les histogrammes
fig, axes1 = plt.subplots(2, 2, figsize=(20, 16))
plt.suptitle("Evolution du nombre de déclaration par saisons entre 2012 et 2025")
plt.subplots_adjust(top=0.90)
ymax_droite = tendance_annuelles.max()*1.1
positions = np.arange(len(tendance_annuelles))
annees = tendance_annuelles.index
for ax, saison in zip(axes1.flat, saisons):
    # 1er axe de gauche : effectifs par saison (les années manquantes à 0)
    effectifs_saison = (
        data.loc[data["SAISONDECL"] == saison, "ANNEEDECL"]
        .value_counts()
        .reindex(tendance_annuelles.index, fill_value=0)
        .sort_index()
    )
    ax.bar(positions, effectifs_saison.values, color='grey',
           label='Effectifs saisonniers')
    ax.set_title(saison)
    ax.set_xlabel("")
    ax.set_xticks(positions)
    ax.set_xticklabels(annees, rotation=45)
    # 2e axe, pour comparer avec la tendance par an
    axe2 = ax.twinx()
    axe2.plot(
        positions,
        tendance_annuelles.values,
        color='red', linestyle='-',
        linewidth=2, label='Effectifs toutes saisons'
    )
    axe2.set_ylim(0, ymax_droite)
    axe2.set_ylabel("Effectif toutes saisons")
    axe2.set_xlim(-0.5, len(tendance_annuelles)-0.5)

    # Légende :
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axe2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left")
# même échelle sur l'axe des ordonnées
ymax = max(ax.get_ylim()[1] for ax in axes1.flat)
for ax in axes1.flat:
    ax.set_ylim(0, ymax)
# Améliorer la lisibilité
axes1[1, 0].set_title("Ete", y=-0.25)
axes1[1, 1].set_title("Automne", y=-0.25)

plt.savefig("Outputs/Evolution du nombre de déclarations par saisons entre 2012 et 2025")
plt.show()
