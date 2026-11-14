import numpy as np
from sklearn.cluster import DBSCAN

RAYON_TERRE_M = 6_371_000


# groupes de points denses, un numéro par groupe et -1 pour le bruit
def regrouper_points(points, eps_metres=100, min_points=5):
    # liste de (lat, lon) en degrés, convertie en radians pour haversine
    coordonnees = np.radians(np.asarray(points, dtype=float))
    if coordonnees.ndim != 2 or coordonnees.shape[1] != 2:
        raise ValueError("points attendus sous la forme [(lat, lon), ...]")

    modele = DBSCAN(
        # rayon en mètres converti en angle sur la sphère
        eps=eps_metres / RAYON_TERRE_M,
        # le point lui-même compte dans ce minimum
        min_samples=min_points,
        # distance réelle sur la terre, pas en degrés
        metric="haversine",
        algorithm="ball_tree",
    )
    return modele.fit_predict(coordonnees)


# petit test avec des points inventés
if __name__ == "__main__":
    points_test = [
        # 4 points proches 
        (48.8590, 2.3470),
        (48.8593, 2.3472),
        (48.8588, 2.3475),
        (48.8591, 2.3467),
        #4 points proches
        (48.8675, 2.3637),
        (48.8677, 2.3640),
        (48.8673, 2.3634),
        # 1 point isolé
        (48.8400, 2.3000),
    ]
    print(regrouper_points(points_test, eps_metres=100, min_points=3))
