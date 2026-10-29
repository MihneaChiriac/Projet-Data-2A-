import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from geodatasets import get_path

data = pd.read_csv("Data/clean_rats.csv")
print(data.X.dtype)
