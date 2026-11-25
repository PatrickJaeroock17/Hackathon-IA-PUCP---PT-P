"""
Modelo de downscaling: la CNN con relieve de train/starter_notebook.ipynb
("5. Baseline 2"), lista para usarse fuera del notebook.

`hour`, `climatology`, `Net`, `to_t`, `prep` y `predict` están copiados del notebook.
La normalización se calcula solo con el split de entrenamiento (< 2024061600) y se
guarda en models/stats.npz.

Uso:
    import modelo
    red = modelo.cargar(static)                  # static = contenido de static.npz
    Y = modelo.predict(red, Xw, times)           # times 'YYYYMMDDHH' en UTC
"""
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

RAIZ = Path(__file__).resolve().parent.parent
RUTA_PESOS = RAIZ / "models" / "modelo.pt"
RUTA_STATS = RAIZ / "models" / "stats.npz"
RUTA_TRAIN = RAIZ / "data" / "train.npz"
INICIO_VAL = "2024061600"

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# Los llena cargar()
GRID = DEM = CLIM = ASD = mu_x = sd_x = None


hour = lambda t: np.array([int(k[-2:]) for k in t])


def climatology(Y, t):
    h = hour(t)
    return np.stack([Y[h == k].mean(0) for k in range(24)])   # (24, 70, 35, 4)


class Net(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(14, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU())
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.dec = nn.Sequential(
            nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, 4, 1))
    def forward(self, x, h):                     # x: (B,3,17,13,4), h: (B,) hora UTC
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
        ang = 2 * np.pi * h / 24
        hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
        f = self.enc(torch.cat([x, hh], 1))
        f = F.grid_sample(f, GRID.expand(B, -1, -1, -1), align_corners=True)   # (B,w,70,35)
        f = torch.cat([f, DEM.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
        return self.dec(f).permute(0, 2, 3, 1)  # (B,70,35,4) anomalía normalizada


def to_t(a): return torch.tensor(a, dtype=torch.float32, device=dev)


def prep(W, t):
    return to_t((W - mu_x) / sd_x), torch.tensor(hour(t), dtype=torch.long, device=dev)


def predict(model, W, t, bs=128):
    model.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(W), bs):
            x, h = prep(W[s:s+bs], t[s:s+bs])
            out.append((CLIM[h] + model(x, h.float()) * ASD).cpu().numpy())
    return np.concatenate(out)


def calcular_stats(ruta_train=RUTA_TRAIN, ruta_stats=RUTA_STATS):
    """Estadísticas del split de entrenamiento, igual que en el notebook."""
    if not Path(ruta_train).exists():
        raise FileNotFoundError(f"No existe {ruta_stats} y falta {ruta_train} para calcularlo.")
    tr = np.load(ruta_train)
    Xtr, Ytr, ttr = tr['X'], tr['Y'], tr['times'].astype(str)
    trn = ~(ttr >= INICIO_VAL)

    mu_x = Xtr[trn].mean((0, 1, 2)); sd_x = Xtr[trn].std((0, 1, 2))
    clim_all = climatology(Ytr[trn], ttr[trn])
    anom_sd = (Ytr[trn] - clim_all[hour(ttr[trn])]).std((0, 1, 2))

    stats = dict(mu_x=mu_x, sd_x=sd_x, clim=clim_all, anom_sd=anom_sd)
    np.savez_compressed(ruta_stats, **stats)
    return stats


def cargar(static, ruta_pesos=RUTA_PESOS, ruta_stats=RUTA_STATS):
    """Prepara la normalización y la grilla, y devuelve la red con los pesos cargados."""
    global GRID, DEM, CLIM, ASD, mu_x, sd_x

    stats = dict(np.load(ruta_stats)) if Path(ruta_stats).exists() else calcular_stats(ruta_stats=ruta_stats)
    mu_x, sd_x = stats['mu_x'], stats['sd_x']
    CLIM = to_t(stats['clim']); ASD = to_t(stats['anom_sd'])

    # Coordenadas de la grilla WRF dentro de la grilla GFS, para grid_sample
    glat, glon = static['gfs_lats'], static['gfs_lons']            # GFS: fila 0 = norte
    wlat, wlon = static['wrf_lats'], static['wrf_lons']            # WRF: fila 0 = sur
    LAT, LON = np.meshgrid(wlat, wlon, indexing='ij')
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    GRID = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32, device=dev)   # (1,70,35,2)
    dem = static['dem']
    DEM = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32, device=dev)[None, None]

    model = Net().to(dev)
    model.load_state_dict(torch.load(ruta_pesos, map_location=dev, weights_only=True))
    model.eval()
    return model
