import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

URL_ARRONDISSEMENTS = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "arrondissements/exports/geojson"
)
CENTRE_PARIS = {"lat": 48.8566, "lon": 2.3522}


# contours des arrondissements, gardés en cache pour pas retélécharger
@st.cache_data
def charger_arrondissements() -> dict:
    reponse = requests.get(URL_ARRONDISSEMENTS, timeout=30)
    # stop si erreur http
    reponse.raise_for_status()
    return reponse.json()


# risque bidon en attendant les vraies données
def creer_risque_factice():
    # seed fixe, mêmes valeurs à chaque lancement
    generateur = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            # numéros en texte, sinon carte sans couleurs
            "arrondissement": [str(numero) for numero in range(1, 21)],
            "risque": generateur.uniform(0, 1, size=20).round(2),
        }
    )


# carte de paris, couleur selon le risque
def tracer_carte(geojson: dict, risques: pd.DataFrame):
    figure = px.choropleth_map( #choropleth = carte avec zones ccolorées
        risques,
        geojson=geojson, #les contouts de l'arrondissment 
        locations="arrondissement",
        # lien tableau / polygones via le numéro d'arrondissement
        featureidkey="properties.c_ar",
        color="risque",
        color_continuous_scale=["white", "red"],
        # échelle fixe, 0 = blanc et 1 = rouge quelles que soient les valeurs
        range_color=(0, 1),
        center=CENTRE_PARIS,
        zoom=10.8,
        opacity=0.75,
        map_style="carto-positron",
        labels={"risque": "Risque", "arrondissement": "Arrondissement"},
    )
    # carte sans marges
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}) #left, right, top, bottom
    return figure


st.title("Risque par arrondissement pour les rats à Paris")
st.warning("Données factices")

carte = tracer_carte(charger_arrondissements(), creer_risque_factice())
st.plotly_chart(carte, width="stretch")
