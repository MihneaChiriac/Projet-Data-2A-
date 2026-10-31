import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from cartiflette import carti_download

data = pd.read_csv("Data/clean_rats.csv")

# Créer la colonne "GEOMETRY" qui comprend des objets Geopandas "shapely", crée à partir de X et Y
# (X et Y étant des coordonnées)
# On vérifie le type des coordonnées
print(data.X.dtype)
# ce sont des float64

geop_data = gpd.GeoDataFrame(
    data, geometry=gpd.points_from_xy(data.X, data.Y), crs="EPSG:4326"
)

print(geop_data.head(5))

# Montrer ces points sur une carte de Paris
# Charger le fond de carte Cartiflette
paris_arrondissements = carti_download(
    crs=4326,
    values=["75"],
    borders="COMMUNE_ARRONDISSEMENT",
    vectorfile_format="geojson",
    filter_by="DEPARTEMENT",
    source="EXPRESS-COG-CARTO-TERRITOIRE",
    year=2022)

# On représente les signalements sur un fond de carte. Pour des raisons de lisibilité,
# on se limite à ............
ax = paris_arrondissements.plot(color="white", edgecolor="black")
geop_data.plot(ax=ax, color="red")
plt.savefig("Outputs/-Total_rats_Paris_arr")
plt.show()