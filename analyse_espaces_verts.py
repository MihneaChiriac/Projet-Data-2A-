import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import osmnx as ox
from scipy import stats
from scipy.spatial import cKDTree
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------
EXCLURE_BOIS = True    # True : retire les bois des observations ET de la référence
TAMPON = 0             # m. >0 agrandit les espaces verts, <0 les rétrécit (test de robustesse)
BW = 10                # largeur des classes de distance (m)
RMAX = 450             # distance maximale affichée (m)
N_REF = 100_000        # nombre de points aléatoires de référence
MIN_N = 20             # effectif minimal d'une classe pour calculer son ratio

rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# ---------------------------------------------------------------
espaces_verts = gpd.read_file('espaces_verts.geojson')    # polygones (EPSG:4326)
data = pd.read_csv('clean_rats_date_clean.csv')

points = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data["X"], data["Y"]),
    crs="EPSG:4326",
)

# ---------------------------------------------------------------
# 2. CONTOUR DE PARIS ET REPROJECTION EN MÈTRES (Lambert 93)
# ---------------------------------------------------------------
paris_gdf = ox.geocode_to_gdf("Paris, France", which_result=1)
print(paris_gdf.geom_type.iloc[0], round(paris_gdf.to_crs(2154).area.iloc[0] / 1e6, 1), "km²")
paris = shapely.make_valid(paris_gdf.to_crs(2154).union_all())

ev = espaces_verts.to_crs(2154)
ev["geometry"] = ev.make_valid()          # répare les géométries invalides
pts_all = points.to_crs(2154)             # toutes les observations (avant filtrage)

# Union de tous les espaces verts, bois compris, y compris les parties qui dépassent
# de Paris : un parc juste après le périph reste un parc proche.
# Elle sert au calcul des distances ET à la définition de "dans un espace vert".
ev_union = ev.union_all()
if TAMPON != 0:
    ev_union = ev_union.buffer(TAMPON)

# ---------------------------------------------------------------
# 3. ZONE D'ÉTUDE ET FILTRAGE DES OBSERVATIONS
# ---------------------------------------------------------------
if EXCLURE_BOIS:
    bois = ev[ev["type_ev"] == "Bois"].union_all()
    zone = shapely.make_valid(paris.difference(bois))
else:
    bois = None
    zone = paris

# Diagnostic : d'où viennent les observations retirées ?
dans_paris = pts_all.intersects(paris).to_numpy()
print(f"{len(pts_all)} observations au total, {(~dans_paris).sum()} hors du polygone de Paris")
if EXCLURE_BOIS:
    dans_bois = (pts_all.intersects(bois).to_numpy()) & dans_paris
    print(f"{dans_bois.sum()} observations dans les bois (retirées)")

# On ne garde que les observations dans la zone d'étude.
# Observations et référence portent ainsi sur la même zone.
pts = pts_all[pts_all.intersects(zone)].copy()
print(f"{len(pts)} observations conservées dans la zone d'étude")

# ---------------------------------------------------------------
# 4. FONCTIONS UTILES
# ---------------------------------------------------------------
def tirer_dans(poly, n, rng):
    """Tire n points uniformément dans un polygone, par rejet :
    tirage dans le rectangle englobant, on garde ceux qui tombent dans le polygone."""
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = [], []
    while sum(len(a) for a in xs) < n:
        x = rng.uniform(minx, maxx, 2 * n)
        y = rng.uniform(miny, maxy, 2 * n)
        m = shapely.contains_xy(poly, x, y)
        xs.append(x[m]); ys.append(y[m])
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]

def ecdf(d, r):
    """Fonction de répartition empirique : pour chaque r, proportion de d <= r."""
    d = np.sort(d)
    return np.searchsorted(d, r, side="right") / len(d)

# ---------------------------------------------------------------
# PARTIE A : PROPORTION D'OBSERVATIONS DANS LES ESPACES VERTS
# ---------------------------------------------------------------

dedans = pts.intersects(ev_union).to_numpy()
n, k = len(pts), int(dedans.sum())

# Part exacte de la surface de la zone couverte par des espaces verts
p0 = zone.intersection(ev_union).area / zone.area


print("\n=== PARTIE A : dans / hors espaces verts ===")
print(f"Observations dans un espace vert : {k}/{n} = {k/n:.1%}")
print(f"Part de la surface en espaces verts : {p0:.1%}")

# ---------------------------------------------------------------
# PARTIE B : OBSERVATIONS HORS DES ESPACES VERTS
# ---------------------------------------------------------------
d_obs_out = pts.distance(ev_union).to_numpy()[~dedans]

# Référence : points uniformes dans la zone privée des espaces verts
zone_hors = shapely.make_valid(zone.difference(ev_union))
xo, yo = tirer_dans(zone_hors, N_REF, rng)
d_ref_out = shapely.distance(shapely.points(xo, yo), ev_union)

print("\n=== PARTIE B : hors espaces verts ===")
print(f"{len(d_obs_out)} observations hors espaces verts")
print("Quantiles 10/25/50/75/90/95 des observations :",
      np.percentile(d_obs_out, [10, 25, 50, 75, 90, 95]).round(0))
print("Quantiles 10/25/50/75/90/95 de la référence  :",
      np.percentile(d_ref_out, [10, 25, 50, 75, 90, 95]).round(0))

# Courbes cumulées conditionnelles
r = np.linspace(0, RMAX, RMAX // 2 + 1)
F_o, F_r = ecdf(d_obs_out, r), ecdf(d_ref_out, r)

# Densités conditionnelles (chacune normalisée par son propre effectif total)
edges = np.arange(0, RMAX + BW, BW)
centres = (edges[:-1] + edges[1:]) / 2
h_o, _ = np.histogram(d_obs_out, bins=edges)
h_r, _ = np.histogram(d_ref_out, bins=edges)
f_o = h_o / len(d_obs_out) / BW
f_r = h_r / len(d_ref_out) / BW

# Ratio, calculé seulement là où les effectifs sont suffisants
valide = (h_o >= MIN_N) & (h_r > 0)
ratio = np.full(len(centres), np.nan)
ratio[valide] = f_o[valide] / f_r[valide]

# --- Bande d'intensité relative maximale ---
i = int(np.nanargmax(ratio))
lo_b, hi_b = edges[i], edges[i + 1]

print(f"\nBande d'intensité relative maximale : {lo_b:.0f}-{hi_b:.0f} m")
print(f"  ratio = {ratio[i]:.2f}  , "
      f"{h_o[i]} observations dans la bande")

# --- Graphiques ---
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].plot(r, F_o, label="Observations hors espaces verts")
ax[0].plot(r, F_r, label="Points aléatoires hors espaces verts")
ax[0].set(xlabel="r (m)", ylabel="Proportion à distance ≤ r", title="Courbes cumulées")
ax[0].legend()

ax[1].plot(centres, f_o * 100, label="Observations")
ax[1].plot(centres, f_r * 100, label="Référence")
ax[1].set(xlabel="r (m)", ylabel="Densité (% par mètre)", title="Densités")
ax[1].legend()

ax[2].plot(centres, ratio)
ax[2].axhline(1, color="grey", ls="--")
ax[2].axvspan(lo_b, hi_b, color="tab:red", alpha=0.2,
              label=f"Max : {lo_b:.0f}-{hi_b:.0f} m (×{ratio[i]:.2f})")
ax[2].set(xlabel="r (m)", ylabel="f_obs / f_ref",
          title="Intensité relative (>1 : sur-représentation)")
ax[2].legend()

for a in ax:
    a.grid(True, alpha=0.3)
    a.minorticks_on()
    a.grid(True, which="minor", alpha=0.1)

plt.savefig("Outputs/hors_espaces_verts.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# GRAPHIQUE RÉSUMÉ DE LA PARTIE A
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.bar(["Observations", "Surface"], [k / n, p0], color=["tab:red", "tab:green"], zorder=3)
ax.set(ylabel="Proportion dans un espace vert", title="Dans un espace vert")
ax.grid(True, axis="y", alpha=0.3, zorder=0)
plt.savefig("Outputs/dans_espaces_verts.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("done espaces verts")
# ===============================================================
# COMPOSTEURS : DISTANCE AU COMPOSTEUR LE PLUS PROCHE
# ===============================================================

RMAX_C = 2000    # m. Les composteurs sont bien plus espacés que les espaces verts
BW_C = 50        # m. Classes plus larges, pour garder des effectifs suffisants

# --- Chargement ---
composteurs = gpd.read_file('dechets-menagers-points-dapport-volontaire-composteurs.geojson')          # <- adaptez le nom du fichier
composteurs = composteurs[~(composteurs.geometry.isna() | composteurs.geometry.is_empty)]
composteurs = composteurs.to_crs(2154)
# Si certaines géométries ne sont pas des points, on prend leur centroïde
composteurs["geometry"] = composteurs.geometry.centroid
print(f"\n{len(composteurs)} composteurs chargés")

# On garde TOUS les composteurs, y compris ceux hors de Paris :
# un composteur juste après le périph reste un composteur proche.
xy_comp = composteurs.geometry.get_coordinates().to_numpy()
arbre = cKDTree(xy_comp)                                    # recherche du plus proche voisin

def dist_au_plus_proche(x, y):
    """Distance (m) de chaque point (x, y) au composteur le plus proche."""
    d, _ = arbre.query(np.column_stack([x, y]))
    return d

# --- Distances des observations ---
xy_obs = pts.geometry.get_coordinates().to_numpy()
d_obs_c = dist_au_plus_proche(xy_obs[:, 0], xy_obs[:, 1])

# --- Référence : points uniformes dans la zone d'étude (même zone que les observations) ---
xc, yc = tirer_dans(zone, N_REF, rng)
d_ref_c = dist_au_plus_proche(xc, yc)

print("\n=== COMPOSTEURS ===")
print(f"{len(d_obs_c)} observations")
print("Quantiles 10/25/50/75/90/95 des observations :",
      np.percentile(d_obs_c, [10, 25, 50, 75, 90, 95]).round(0))
print("Quantiles 10/25/50/75/90/95 de la référence  :",
      np.percentile(d_ref_c, [10, 25, 50, 75, 90, 95]).round(0))

# --- Courbes cumulées ---
r_c = np.linspace(0, RMAX_C, RMAX_C // 5 + 1)
F_o_c, F_r_c = ecdf(d_obs_c, r_c), ecdf(d_ref_c, r_c)

# --- Densités (chacune normalisée par son propre effectif total) ---
edges_c = np.arange(0, RMAX_C + BW_C, BW_C)
centres_c = (edges_c[:-1] + edges_c[1:]) / 2
h_o_c, _ = np.histogram(d_obs_c, bins=edges_c)
h_r_c, _ = np.histogram(d_ref_c, bins=edges_c)
f_o_c = h_o_c / len(d_obs_c) / BW_C
f_r_c = h_r_c / len(d_ref_c) / BW_C

# Ratio, seulement là où les effectifs sont suffisants
valide_c = (h_o_c >= MIN_N) & (h_r_c > 0)
ratio_c = np.full(len(centres_c), np.nan)
ratio_c[valide_c] = f_o_c[valide_c] / f_r_c[valide_c]

# --- Bande d'intensité relative maximale ---
i_c = int(np.nanargmax(ratio_c))
lo_c, hi_c = edges_c[i_c], edges_c[i_c + 1]
print(f"\nBande d'intensité relative maximale : {lo_c:.0f}-{hi_c:.0f} m")
print(f"  ratio = {ratio_c[i_c]:.2f}, {h_o_c[i_c]} observations dans la bande")

# --- Graphiques ---
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].plot(r_c, F_o_c, label="Observations")
ax[0].plot(r_c, F_r_c, label="Points aléatoires dans la zone")
ax[0].set(xlabel="r (m)", ylabel="Proportion à distance ≤ r d'un composteur",
          title="Courbes cumulées")
ax[0].legend()

ax[1].plot(centres_c, f_o_c * 100, label="Observations")
ax[1].plot(centres_c, f_r_c * 100, label="Référence")
ax[1].set(xlabel="r (m)", ylabel="Densité (% par mètre)", title="Densités")
ax[1].legend()

ax[2].plot(centres_c, ratio_c)
ax[2].axhline(1, color="grey", ls="--")
ax[2].axvspan(lo_c, hi_c, color="tab:red", alpha=0.2,
              label=f"Max : {lo_c:.0f}-{hi_c:.0f} m (×{ratio_c[i_c]:.2f})")
ax[2].set(xlabel="r (m)", ylabel="f_obs / f_ref",
          title="Intensité relative (>1 : sur-représentation)")
ax[2].legend()

for a in ax:
    a.grid(True, alpha=0.3)
    a.minorticks_on()
    a.grid(True, which="minor", alpha=0.1)

plt.savefig("Outputs/composteurs.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("done composteurs")

# Distance aux espaces verts pour observations et référence (mêmes points que d_obs_c / d_ref_c)
d_ev_obs = pts.distance(ev_union).to_numpy()
d_ev_ref = shapely.distance(shapely.points(xc, yc), ev_union)

# --- Strates : observations ---
dedans_obs = pts.intersects(ev_union).to_numpy()

# --- Strates : référence (points déjà tirés dans zone : xc, yc) ---
pts_ref = shapely.points(xc, yc)
dedans_ref = shapely.intersects(pts_ref, ev_union)      # d_ev_ref == 0 équivaut à ça

# ===============================================================
# STRATE 1 : DANS LES ESPACES VERTS (hors bois si EXCLURE_BOIS)
# ===============================================================
d_in_o = d_obs_c[dedans_obs]
d_in_r = d_ref_c[dedans_ref]
print(f"Dans espaces verts : {dedans_obs.sum()} obs., {dedans_ref.sum()} points de référence")

# Panneau de gauche : courbes cumulées
r_in = np.linspace(0, RMAX_C, RMAX_C // 5 + 1)

# Panneau de droite : bandes de 250 m
BW_IN = 250
edges_in = np.arange(0, RMAX_C + BW_IN, BW_IN)
h_o, _ = np.histogram(d_in_o, bins=edges_in)
h_r, _ = np.histogram(d_in_r, bins=edges_in)
ratio_in = np.full(len(h_o), np.nan)
v = (h_o >= MIN_N) & (h_r > 0)
ratio_in[v] = (h_o[v] / len(d_in_o)) / (h_r[v] / len(d_in_r))

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
ax[0].plot(r_in, ecdf(d_in_o, r_in), label="Observations dans espaces verts")
ax[0].plot(r_in, ecdf(d_in_r, r_in), label="Référence dans espaces verts")
ax[0].set(xlabel="r (m)", ylabel="Proportion à distance ≤ r d'un composteur", title="Courbes cumulées")
ax[0].legend()

ax[1].bar(edges_in[:-1], np.nan_to_num(ratio_in), width=BW_IN, align="edge",
          edgecolor="k", alpha=0.7)
ax[1].axhline(1, color="grey", ls="--")
# Nombre d'observations au-dessus de chaque bande (y compris celles masquées, effectif < MIN_N)
ymax = np.nanmax(ratio_in) if np.isfinite(ratio_in).any() else 1
for x0, n_b, rt in zip(edges_in[:-1], h_o, ratio_in):
    y = rt if np.isfinite(rt) else 0
    ax[1].text(x0 + BW_IN / 2, y + 0.02 * ymax, f"n = {n_b}", ha="center", va="bottom", fontsize=8)
ax[1].set_xticks(edges_in)
ax[1].set(xlabel="Distance au composteur (m)", ylabel="f_obs / f_ref",
          title="Intensité relative (dans les espaces verts), bandes de 250 m")

for a in ax:
    a.grid(True, alpha=0.3)
plt.savefig("Outputs/composteurs_dans_ev.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===============================================================
# STRATE 2 : HORS ESPACES VERTS (carte 2D, quantiles de la strate)
# ===============================================================
dev_o = d_ev_obs[~dedans_obs]; dc_o = d_obs_c[~dedans_obs]
dev_r = d_ev_ref[~dedans_ref]; dc_r = d_ref_c[~dedans_ref]
print(f"Hors espaces verts : {len(dev_o)} obs., {len(dev_r)} points de référence")

rho_out = stats.spearmanr(dev_r, dc_r)[0]
print("Spearman (référence hors espaces verts) :", rho_out.round(2))

u_r, u_o = ecdf(dev_r, dev_r), ecdf(dev_r, dev_o)
v_r, v_o = ecdf(dc_r, dc_r), ecdf(dc_r, dc_o)

b = np.linspace(0, 1, 11)
H_o, _, _ = np.histogram2d(u_o, v_o, bins=[b, b])
H_r, _, _ = np.histogram2d(u_r, v_r, bins=[b, b])
rat = np.full(H_o.shape, np.nan)
ok = (H_o >= MIN_N) & (H_r > 0)
rat[ok] = (H_o[ok] / H_o.sum()) / (H_r[ok] / H_r.sum())

def graduations2(ax, dx, dy, pas=0.2, fontsize=8):
    q = np.round(np.arange(0, 1 + 1e-9, pas), 2)
    ax.set_xticks(q); ax.set_xticklabels([f"{x:.1f}\n{np.quantile(dx, x):.0f} m" for x in q], fontsize=fontsize)
    ax.set_yticks(q); ax.set_yticklabels([f"{x:.1f} · {np.quantile(dy, x):.0f} m" for x in q], fontsize=fontsize)

fig, ax = plt.subplots(1, 3, figsize=(22, 6), constrained_layout=True)
sel = np.random.default_rng(1).choice(len(u_r), size=min(len(u_o), len(u_r)), replace=False)
ax[0].scatter(u_r[sel], v_r[sel], s=2, alpha=0.3)
ax[0].set_title(f"Référence hors espaces verts (Spearman = {rho_out:.2f})")
ax[1].scatter(u_o, v_o, s=2, alpha=0.3, color="tab:red")
ax[1].set_title("Observations hors espaces verts")
for a in ax[:2]:
    a.set(xlabel="Quantile (et distance) à l'espace vert", ylabel="Quantile (et distance) au composteur",
          xlim=(0, 1), ylim=(0, 1))
    graduations2(a, dev_r, dc_r)
    a.grid(True, alpha=0.3)

m = ax[2].pcolormesh(b, b, rat.T, cmap="RdBu_r", norm=LogNorm(vmin=0.2, vmax=5), shading="flat")
plt.colorbar(m, ax=ax[2], label="Intensité relative (1 = hasard)")
ax[2].set(xlabel="Quantile (et distance) à l'espace vert", ylabel="Quantile (et distance) au composteur",
          title="Intensité relative (cases de même masse)")
graduations2(ax[2], dev_r, dc_r, pas=0.1, fontsize=7)
plt.savefig("Outputs/rangs_2d_hors_ev.png", dpi=150, bbox_inches="tight")
plt.close(fig)