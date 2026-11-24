# Rat-dar

La légende urbaine veut que plus de 5 millions de rats vivent à Paris. Pour limiter la prolifération des rongeurs, la Ville de Paris s'appuie notamment sur l'application Dans Ma Rue, qui permet à chacun de signaler la présence de rats dans l'espace public.

Ce projet cherche à savoir s'il est possible de « prédire » la présence d'un rat à un moment donné, à un lieu donné. Aux dates et aux coordonnées GPS des signalements de rats de Dans Ma Rue entre 2012 et 2025, nous ajoutons des données météorologiques et géographiques adaptées à l'espace urbain parisien : surfaces d'eau de l'IGN, espaces verts, composteurs, grille de carreaux de 200 m. À elles seules, ces variables suffisent-elles à prédire le prochain signalement de rat ?

## Lire le projet

Toute l'analyse se trouve dans le notebook `notebook_corrigé.ipynb`, à lire dans l'ordre. Il présente les fichiers du dépôt au début.

## Tester l'application

Depuis la racine du dépôt, installer les dépendances (testé avec Python 3.12), puis lancer l'application Streamlit :

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

L'application s'ouvre dans le navigateur. Elle lit `données/clean_rats.csv` et `données/clean_meteo.csv`, et a besoin d'une connexion internet pour la météo du jour et les contours des arrondissements.
