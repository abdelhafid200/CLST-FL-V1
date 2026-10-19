# CLST-FL : EXP-Next, attribution du gate (sélectivité contrôlée)
# K=2 pour toutes les méthodes (sauf l'oracle). Modèle, données et splits inchangés ; seeds 123-127.
# Conditions : FT (réf) | compat_k2 | compat_k2_small | compat_count_shuffled | sel@1.645 | sel@2.0 | oracle_truth_gate
# 1.645 et 2.0 sont des seuils opérationnels fixés à l'avance, pas un niveau alpha=0.05 au sens d'une e-value.
# N'écrit aucun fichier, les résultats sont seulement affichés.
import os, json, pickle, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.stats import spearmanr
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config
LEVEL = "high"
BASE_DIR = "data_v3"   # dossier écrit par data_00_build_clients_v3.py ; à adapter
CLIENT_DIR = os.path.join(BASE_DIR, f"clients_hetero_{LEVEL}")
SEEDS = [123, 124, 125, 126, 127]
N_ROUNDS, WARMUP, MEAS_EVERY = 50, 3, 5
MEAS_ROUNDS = list(range(WARMUP + MEAS_EVERY, N_ROUNDS + 1, MEAS_EVERY))
LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE = 5, 1e-3, 64
HIDDEN, DROPOUT = (64, 32), 0.10
K, GAMMA = 2, 0.10                     # même K pour toutes les méthodes
TC_P, TC_S = 1.645, 2.0                # gate proposé / gate conservateur
HARM_DELTA, BOOT_B, CVAR_Q = 1e-4, 2000, 0.10
EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS = ["layer1", "layer2", "head"]; HIDL = ["layer1", "layer2"]

with open(os.path.join(CLIENT_DIR, "metadata.json"), "r") as f:
    META = json.load(f)
INPUT_DIM = META["input_dim"]
TRUE = np.array(META["true_compat"]) if "true_compat" in META else None

PROPOSED, STRICT = f"sel@{TC_P}", f"sel@{TC_S}"
RULES = ["compat_k2", "compat_k2_small", "compat_count_shuffled", PROPOSED, STRICT]
if TRUE is not None: RULES.append("oracle_truth_gate")
RIDX = {n: i for i, n in enumerate(RULES)}; nR = len(RULES)

# Modèle
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]; prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

assert len(HIDDEN) == 2
_scratch = SimpleMLP().cpu().eval()
_SHAPES = [tuple(p.shape) for p in SimpleMLP().parameters()]
_SIZES = [int(np.prod(s)) for s in _SHAPES]; P = sum(_SIZES)

@torch.no_grad()
def probs_of(vec, X):
    vector_to_parameters(vec.clone(), _scratch.parameters()); _scratch.eval()
    return torch.sigmoid(_scratch(X)).numpy()

@torch.no_grad()
def batch_probs(V, X):
    m, o, parts = V.shape[0], 0, []
    for s, sz in zip(_SHAPES, _SIZES):
        parts.append(V[:, o:o + sz].reshape(m, *s)); o += sz
    W1, b1, W2, b2, W3, b3 = parts
    h = torch.relu(torch.einsum('nd,mhd->mnh', X, W1) + b1[:, None, :])
    h = torch.relu(torch.einsum('mnh,mgh->mng', h, W2) + b2[:, None, :])
    return torch.sigmoid(torch.einsum('mng,mog->mno', h, W3).squeeze(-1) + b3).numpy()

def train_local(start_vec, cid, seed, epochs=LOCAL_EPOCHS):
    torch.manual_seed(int(seed))
    model = SimpleMLP().to(DEVICE)
    vector_to_parameters(start_vec.clone().to(DEVICE), model.parameters()); model.train()
    X, y = data[cid]["Xtr_dev"], data[cid]["ytr_dev"]
    opt = torch.optim.Adam(model.parameters(), lr=LOCAL_LR); n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]; opt.zero_grad()
            ((torch.sigmoid(model(X[idx])) - y[idx]) ** 2).mean().backward(); opt.step()
    return parameters_to_vector(model.parameters()).detach().cpu().clone()

def fedavg(vecs): return sum((n_train[k] / N_TOTAL) * vecs[k] for k in IDS)

def pmask(idxs):
    m, o = torch.zeros(P), 0
    for i, sz in enumerate(_SIZES):
        if i in idxs: m[o:o + sz] = 1.0
        o += sz
    return m
CMASK = dict(layer1=pmask({0, 1}), layer2=pmask({2, 3}), head=pmask({4, 5}))

# Données
def T(x, dev="cpu"): return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
clients = {}
for fname in sorted(os.listdir(CLIENT_DIR)):
    if fname.startswith("client_") and fname.endswith(".pkl"):
        with open(os.path.join(CLIENT_DIR, fname), "rb") as f: c = pickle.load(f)
        clients[c["client_id"]] = c
IDS = sorted(clients); nK = len(IDS)
assert all("X_test" in clients[k] for k in IDS), "X_test absent"
n_train = {cid: len(clients[cid]["y_train"]) for cid in IDS}; N_TOTAL = sum(n_train.values())
NTR = np.array([n_train[k] for k in IDS])
PREV = np.array([np.mean(clients[k]["y_train"]) for k in IDS])
SIZEQ = pd.qcut(pd.Series(NTR).rank(method="first"), 4, labels=False).values
PREVQ = pd.qcut(pd.Series(PREV).rank(method="first"), 4, labels=False).values
SMALL = NTR <= np.median(NTR)

data = {}
for cid in IDS:
    c = clients[cid]
    data[cid] = dict(Xtr_dev=T(c["X_train"], DEVICE), ytr_dev=T(c["y_train"], DEVICE),
                     Xva=T(c["X_val"]), yva=np.asarray(c["y_val"], dtype=float),
                     Xte=T(c["X_test"]), yte=np.asarray(c["y_test"], dtype=float))
print(f"level={LEVEL} | {nK} clients | n_train {NTR.min()}/{int(np.median(NTR))}/{NTR.max()} | "
      f"prev {PREV.min():.3f}/{np.median(PREV):.3f}/{PREV.max():.3f} | params={P} | K={K} | "
      f"true_compat: {'oui' if TRUE is not None else 'NON (oracle désactivé)'} | règles: {RULES}")

def make_splits():
    sel = {cid: (data[cid]["Xva"], data[cid]["yva"]) for cid in IDS}      # sélection = val complète
    ev = {cid: (data[cid]["Xte"], data[cid]["yte"]) for cid in IDS}       # évaluation = test
    return sel, ev

def topk_mask(score, allowed, k=K):
    out = np.zeros(len(allowed), bool); idx = np.where(allowed)[0]
    if len(idx) and k > 0: out[idx[np.argsort(-score[idx])[:k]]] = True
    return out

# Mesure
def new_acc():
    z = lambda: np.zeros((nR, nK))
    return dict(SB=z(), ACC=z(), POS=z(), HARM=z(), NSRC=z(), COV1=z(), COV2=z(), FTB=np.zeros(nK), M=0)

def measure(FTt, Gprev, sel, ev, acc, seed):
    acc["M"] += 1
    Dt = FTt - Gprev[None]
    V = {l: Dt * CMASK[l] for l in HIDL}
    info = {}
    # passe 1 : top-K par compat, gate, gains par paire sur EVAL
    for ki, k in enumerate(IDS):
        Xs, ys = sel[k]; Xe, ye = ev[k]; ns, ne = len(ys), len(ye)
        if ns < 5 or ne == 0: continue
        ys = np.asarray(ys, float); ye = np.asarray(ye, float)
        Xc, yc = torch.cat([Xs, Xe]), np.concatenate([ys, ye])
        Lb = (batch_probs(FTt[ki][None], Xc)[0] - yc) ** 2
        valid = np.arange(nK) != ki
        CL = ((batch_probs(FTt, Xs) - ys[None]) ** 2).mean(1)
        cp = CL[ki] - CL; cp[ki] = -np.inf                       # >0 : j prédit mieux que k
        cand = topk_mask(cp, valid, K)                           # top-2 par compatibilité
        gE, gate = {}, {}
        for l in HIDL:
            Pm = batch_probs(FTt[ki][None] + GAMMA * V[l], Xc)
            Dd = Lb[None] - (Pm - yc[None]) ** 2; ds = Dd[:, :ns]
            g = ds.mean(1); t = np.nan_to_num(g / (ds.std(1, ddof=1) / np.sqrt(ns) + EPS), nan=-9.0)
            ge = Dd[:, ns:].mean(1); ge[ki] = np.nan; gE[l] = ge
            for tc in (TC_P, TC_S): gate[(l, tc)] = valid & (t > tc) & (g > 0)
        info[ki] = dict(cp=cp, cand=cand, valid=valid, gE=gE, gate=gate, Xe=Xe, ye=ye, Lb_e=float(Lb[ns:].mean()))
    # comptes du gate, puis permutés entre clients (même nombre de sources, mais la sélection est cassée)
    rng = np.random.default_rng(10_000 * seed + acc["M"])
    cnt = np.zeros((nK, len(HIDL)), int)
    for ki, d in info.items():
        for li, l in enumerate(HIDL): cnt[ki, li] = int((d["cand"] & d["gate"][(l, TC_P)]).sum())
    perm = rng.permutation(nK); zero = np.zeros(nK, bool)
    # passe 2 : plans de transfert et évaluation
    for ki, d in info.items():
        cand, cp, valid, gate = d["cand"], d["cp"], d["valid"], d["gate"]
        plans = {
            "compat_k2": {l: cand for l in HIDL},
            "compat_k2_small": {l: (cand if SMALL[ki] else zero) for l in HIDL},
            "compat_count_shuffled": {l: topk_mask(cp, valid, k=int(cnt[perm[ki], li])) for li, l in enumerate(HIDL)},
            PROPOSED: {l: cand & gate[(l, TC_P)] for l in HIDL},
            STRICT: {l: cand & gate[(l, TC_S)] for l in HIDL}}
        if TRUE is not None:
            ct = topk_mask(TRUE[ki], valid, K)
            plans["oracle_truth_gate"] = {l: ct & gate[(l, TC_P)] for l in HIDL}
        vecs = []
        for n in RULES:
            v = FTt[ki].clone()
            for l, m in plans[n].items():
                if m.any(): v = v + GAMMA * (torch.tensor(m / m.sum(), dtype=torch.float32) @ V[l])
            vecs.append(v)
        Pr = batch_probs(torch.stack(vecs), d["Xe"])
        acc["FTB"][ki] += d["Lb_e"]
        for i, n in enumerate(RULES):
            ri = RIDX[n]; acc["SB"][ri, ki] += d["Lb_e"] - ((Pr[i] - d["ye"]) ** 2).mean()
            masks = [plans[n][l] for l in HIDL]
            for l, m in plans[n].items():
                g = d["gE"][l][m]
                acc["ACC"][ri, ki] += m.sum(); acc["POS"][ri, ki] += (g > 0).sum(); acc["HARM"][ri, ki] += (g < -HARM_DELTA).sum()
            u = int(np.logical_or.reduce(masks).sum())
            acc["NSRC"][ri, ki] += np.mean([m.sum() for m in masks]); acc["COV1"][ri, ki] += (u >= 1); acc["COV2"][ri, ki] += (u >= 2)

def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    G = parameters_to_vector(SimpleMLP().parameters()).detach().cpu().clone()
    sel, ev = make_splits(); acc, t0 = new_acc(), time.time()
    for r in range(1, N_ROUNDS + 1):
        FT = {k: train_local(G, k, seed + r * 1000 + k) for k in IDS}
        Gnew = fedavg(FT)
        if r in MEAS_ROUNDS: measure(torch.stack([FT[k] for k in IDS]), G, sel, ev, acc, seed)
        G = Gnew
    print(f"[seed={seed}] terminé en {(time.time() - t0) / 60:.1f} min")
    return acc

_V = torch.stack([parameters_to_vector(SimpleMLP().parameters()).detach() for _ in range(3)]); _X = data[IDS[0]]["Xva"][:64]
assert np.allclose(batch_probs(_V, _X)[1], probs_of(_V[1], _X), atol=1e-5), "batch_probs != probs_of"
t00 = time.time(); ACCS = [run_seed(s) for s in SEEDS]

# Analyse
SM = sum(a["M"] for a in ACCS); agg = lambda key: sum(a[key] for a in ACCS)
dB = agg("SB") / SM; FTB = agg("FTB") / SM                                   # dBrier par client (>0 = mieux que FT)
ACC_, POS_, HARM_ = agg("ACC"), agg("POS"), agg("HARM")
NSRC, COV1, COV2 = agg("NSRC") / SM, agg("COV1") / SM, agg("COV2") / SM
HF = np.where(ACC_ > 0, HARM_ / np.maximum(ACC_, 1), 0.0)                    # fraction de paires nuisibles par client (0 si aucun transfert)
HE = HARM_ / (SM * len(HIDL) * K)                                            # paires nuisibles / budget K=2 (exposition)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)

def cvar(v, q=CVAR_Q):
    v = np.sort(np.asarray(v, float)); m = max(1, int(np.ceil(q * len(v)))); return v[:m].mean()
def boot(fn, pool=None, B=BOOT_B, seed=1):
    pool = np.arange(nK) if pool is None else np.asarray(pool)
    rg = np.random.default_rng(seed)
    bs = np.array([fn(pool[rg.integers(0, len(pool), len(pool))]) for _ in range(B)])
    return fn(pool), *np.percentile(bs, [2.5, 97.5])
def fm(t, s=1e3, nd=3): return f"{t[0]*s:+.{nd}f} [{t[1]*s:+.{nd}f}, {t[2]*s:+.{nd}f}]"

print("=" * 140 + f"\nA1. PERFORMANCE PAR CLIENT [{LEVEL}] seeds={SEEDS} K={K} | FT Brier moyen={FTB.mean():.4f} | dBrier x1e3 (>0 = mieux que FT_k)\n" + "=" * 140)
rows = [dict(rule="FT (réf)", dB_mean="+0.000", P10="0", P25="0", P50="0", P75="0", P90="0", CVaR10="0", worst="0", per_seed="0")]
for r in RULES:
    v = dB[RIDX[r]]; ps = [float((a["SB"][RIDX[r]] / a["M"]).mean()) for a in ACCS]
    rows.append(dict(rule=r, dB_mean=fm(boot(lambda i: v[i].mean())),
                     **{f"P{q}": f"{np.percentile(v, q)*1e3:+.3f}" for q in (10, 25, 50, 75, 90)},
                     CVaR10=fm(boot(lambda i: cvar(v[i]))), worst=f"{v.min()*1e3:+.3f}", per_seed="/".join(f"{x*1e3:+.2f}" for x in ps)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 140 + "\nA2. SÉLECTION ET QUALITÉ DES TRANSFERTS\n" + "=" * 140)
rows = []
for r in RULES:
    i = RIDX[r]; a = ACC_[i].sum()
    rows.append(dict(rule=r, n_src_mean=round(NSRC[i].mean(), 3), n_src_median=round(float(np.median(NSRC[i])), 3),
                     coverage_ge1=round(COV1[i].mean(), 3), coverage_ge2=round(COV2[i].mean(), 3),
                     precision_pair=round(POS_[i].sum() / a, 3) if a > 0 else np.nan,
                     harm_pair=round(HARM_[i].sum() / a, 3) if a > 0 else np.nan,
                     harm_frac_client=fm(boot(lambda j: HF[i][j].mean()), 1, 3), harm_exposure=fm(boot(lambda j: HE[i][j].mean()), 1, 3)))
print(pd.DataFrame(rows).to_string(index=False))

# B. contrastes appariés (clients)
def contrast(a, b, pool=None):
    ia, ib = RIDX[a], RIDX[b]; d = dB[ia] - dB[ib]
    return dict(mean=boot(lambda i: d[i].mean(), pool), cvar_delta=boot(lambda i: cvar(d[i]), pool),
                dcvar=boot(lambda i: cvar(dB[ia][i]) - cvar(dB[ib][i]), pool),
                harm=boot(lambda i: (HF[ia] - HF[ib])[i].mean(), pool), hexp=boot(lambda i: (HE[ia] - HE[ib])[i].mean(), pool),
                seeds_pos=sum(float(((x["SB"][ia] - x["SB"][ib]).mean() / x["M"])) > 0 for x in ACCS))

print("\n" + "=" * 140 + "\nB. CONTRASTES APPARIÉS (IC95 bootstrap sur les clients) — dBrier x1e3 ; harm en fraction\n"
      "   CVaR10(Δ) = CVaR10 de la différence par client | ΔCVaR10 = CVaR10(a) - CVaR10(b) | harm : a - b (négatif = a moins nuisible)\n" + "=" * 140)
PAIRS = [(PROPOSED, "compat_k2"), (PROPOSED, "compat_count_shuffled"), (PROPOSED, "compat_k2_small"), (STRICT, "compat_k2"), (PROPOSED, STRICT)]
if TRUE is not None: PAIRS.append(("oracle_truth_gate", PROPOSED))
rows = []
for a, b in PAIRS:
    c = contrast(a, b)
    rows.append(dict(contraste=f"{a} - {b}", mean=fm(c["mean"]), CVaR10_delta=fm(c["cvar_delta"]), dCVaR10=fm(c["dcvar"]),
                     harm_frac=fm(c["harm"], 1, 3), harm_expo=fm(c["hexp"], 1, 3), seeds_pos=f"{c['seeds_pos']}/{len(SEEDS)}"))
print(pd.DataFrame(rows).to_string(index=False))

# C. stratification (taille puis prévalence)
def strat(tier, title):
    print("\n" + "=" * 140 + f"\nC. STRATIFICATION PAR {title} (Q1 = plus petit/faible, Q4 = plus grand/élevé) — CVaR10 intra-tier = moyenne des ceil(10%) pires clients du tier\n" + "=" * 140)
    cols = ["FT"] + RULES; out = {k: [] for k in ("dBrier x1e3 (moyenne)", "CVaR10 x1e3", "harm_frac", "coverage>=1")}; idx = []
    for q in range(4):
        pool = np.where(tier == q)[0]; idx.append(f"Q{q+1} (n={len(pool)})")
        out["dBrier x1e3 (moyenne)"].append(["0.000"] + [f"{dB[RIDX[r]][pool].mean()*1e3:+.3f}" for r in RULES])
        out["CVaR10 x1e3"].append(["0.000"] + [f"{cvar(dB[RIDX[r]][pool])*1e3:+.3f}" for r in RULES])
        out["harm_frac"].append(["-"] + [f"{HF[RIDX[r]][pool].mean():.3f}" for r in RULES])
        out["coverage>=1"].append(["-"] + [f"{COV1[RIDX[r]][pool].mean():.3f}" for r in RULES])
    for name, M in out.items():
        print(f"\n-- {name}"); print(pd.DataFrame(M, index=idx, columns=cols).to_string())
    print(f"\n-- FT Brier moyen par tier : " + " | ".join(f"Q{q+1}={FTB[tier == q].mean():.4f}" for q in range(4)))
    print(f"\n-- Contraste {PROPOSED} - compat_k2 PAR TIER (IC95 intra-tier)")
    rows = []
    for q in range(4):
        pool = np.where(tier == q)[0]; c = contrast(PROPOSED, "compat_k2", pool); c2 = contrast(PROPOSED, "compat_count_shuffled", pool)
        rows.append(dict(tier=f"Q{q+1}", dB_vs_k2=fm(c["mean"]), harm_vs_k2=fm(c["harm"], 1, 3),
                         dB_vs_shuffled=fm(c2["mean"]), harm_vs_shuffled=fm(c2["harm"], 1, 3)))
    print(pd.DataFrame(rows).to_string(index=False))
strat(SIZEQ, "TAILLE (n_train)")
strat(PREVQ, "PRÉVALENCE")

print("\n" + "=" * 140 + "\nD. LE GAIN DU GATE DÉPEND-IL DU CLIENT ? Spearman (par client)\n" + "=" * 140)
rows = []
for b in ("compat_k2", "compat_count_shuffled", "compat_k2_small"):
    d = dB[RIDX[PROPOSED]] - dB[RIDX[b]]; dh = HF[RIDX[PROPOSED]] - HF[RIDX[b]]
    r1, p1 = spearmanr(d, np.log(NTR)); r2, p2 = spearmanr(d, PREV); r3, p3 = spearmanr(dh, np.log(NTR))
    rows.append(dict(contraste=f"{PROPOSED} - {b}", dB_vs_log_ntrain=f"{r1:+.3f} (p={p1:.3f})", dB_vs_prevalence=f"{r2:+.3f} (p={p2:.3f})",
                     dHarm_vs_log_ntrain=f"{r3:+.3f} (p={p3:.3f})"))
print(pd.DataFrame(rows).to_string(index=False))

# E. verdict
print("\n" + "=" * 140 + "\nE. VERDICT — Gate Attribution (K=2 contrôlé)\n" + "=" * 140)
c = contrast(PROPOSED, "compat_k2"); c3 = contrast(PROPOSED, "compat_count_shuffled")
g1 = c["cvar_delta"][1] > 0; g2 = c["harm"][2] < 0; g3 = (c3["cvar_delta"][1] > 0) or (c3["harm"][2] < 0)
show = lambda name, ok, info: print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ->  {info}")
show(f"H1a  CVaR10({PROPOSED} - compat_k2) > 0 (IC95 > 0)", g1, fm(c["cvar_delta"]))
show(f"H1b  harm_frac({PROPOSED}) - harm_frac(compat_k2) < 0 (IC95 < 0)", g2, fm(c["harm"], 1, 3))
show(f"H2   vs compat_count_shuffled : CVaR10(Δ) > 0 OU harm < 0 (au-delà du simple nombre de sources)", g3,
     f"CVaR10(Δ) {fm(c3['cvar_delta'])} | harm {fm(c3['harm'], 1, 3)}")
print(f"\n  Gain moyen {PROPOSED} - compat_k2 : {fm(c['mean'])}   |   {PROPOSED} - FT : {fm(boot(lambda i: dB[RIDX[PROPOSED]][i].mean()))}")
if g1 and g2:
    print("  => GATE CONFIRMÉ : protection supplémentaire au-delà de la sélection de 2 partenaires. Garder l'e-process.")
elif g1 or g2:
    print("  => PARTIEL : un seul des deux critères (CVaR10 / harm) est confirmé. Garder le gate comme garde-fou, claim prudent ; vérifier avec plus de seeds / la strate petits clients.")
else:
    print("  => NON CONFIRMÉ : sel ≈ compat_k2. Retirer l'e-process du cœur de l'algorithme ; papier centré sur compatibility-driven sparse client-layer transfer.")
print(f"\nTotal time: {(time.time() - t00) / 60:.1f} min")
