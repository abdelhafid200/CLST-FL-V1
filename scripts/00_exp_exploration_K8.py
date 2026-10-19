# CLST-FL : run avec K=8 (scénario B, sans clusters), sur la trajectoire FedAvg. Le test ne sert qu'à mesurer.
# Prévu au départ comme confirmatoire ; dans l'article il est présenté comme exploration (voir le README).
# À lancer deux fois en ne changeant que LEVEL : "iid" (contrôle N0), puis "high".
# Fixés avant le run : PRIMARY_TC=1.645, GAMMA=0.1, K=8, nouvelles seeds ; sélection sur toute la val, évaluation sur le test.
# Règles : fedavg_global | self_extrap | uniform | sim | compat | compat_lw | lw_cos | compat_rand_matched | sel@t | sel_sim | sel_all
#          | oracle_truth, oracle_truth_gate (uniquement si META contient true_compat ; ce sont des références, pas des méthodes)
import os, json, pickle, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.stats import spearmanr
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config
LEVEL = "high"                                   # "iid" (N0) puis "high"
BASE_DIR = "data_v3"   # dossier écrit par data_00_build_clients_v3.py ; à adapter
CLIENT_DIR = os.path.join(BASE_DIR, f"clients_hetero_{LEVEL}")
RESULTS_DIR = "results_confirm"; os.makedirs(RESULTS_DIR, exist_ok=True)
SEEDS = [123, 124, 125, 126, 127]                # pas encore utilisées (exploration : 42-44) ; ~10 min par seed, passer à 3 si c'est trop long
N_ROUNDS, WARMUP, MEAS_EVERY = 50, 3, 5
MEAS_ROUNDS = list(range(WARMUP + MEAS_EVERY, N_ROUNDS + 1, MEAS_EVERY))
LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE = 5, 1e-3, 64
HIDDEN, DROPOUT = (64, 32), 0.10
K_NEIGHBORS, GAMMA = 8, 0.10
TC_GRID = [0.0, 1.0, 1.645, 2.0]                 # frontière (exploratoire)
PRIMARY_TC = 1.645                               # fixé avant le run
SEL_FRAC, HARM_DELTA, BOOT_B = 0.50, 1e-4, 2000  # SEL_FRAC inutilisé quand USE_TEST=True
USE_TEST = True
BALANCE_BLOCKS, EPS = True, 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS = ["layer1", "layer2", "head"]; HIDL = ["layer1", "layer2"]

with open(os.path.join(CLIENT_DIR, "metadata.json"), "r") as f:
    META = json.load(f)
INPUT_DIM = META["input_dim"]
TRUE = np.array(META["true_compat"]) if "true_compat" in META else None

PROPOSED = f"sel@{PRIMARY_TC}"
RULES = ["fedavg_global", "self_extrap", "uniform", "sim", "compat", "compat_lw", "lw_cos", "compat_rand_matched"]
RULES += [f"sel@{t}" for t in TC_GRID] + [f"sel_sim@{PRIMARY_TC}", f"sel_all@{PRIMARY_TC}"]
if TRUE is not None: RULES += ["oracle_truth", "oracle_truth_gate"]
# RULES += ["compat_k1", "compat_k2", "compat_count_shuffled"]
N_PERM = 20
RULES += ["compat_k1", "compat_k2", "compat_count_shuffled"]
RULES += ["compat_k2_small", "compat_k2_gatewhether"] + [f"shuf20_{j}" for j in range(N_PERM)]
N_SHUFS = 10
RULES += [f"shufS_{j}" for j in range(N_SHUFS)]
RIDX = {n: i for i, n in enumerate(RULES)}; nR = len(RULES)
RIDX = {n: i for i, n in enumerate(RULES)}; nR = len(RULES)
NOPAIR = ("fedavg_global", "self_extrap")
LW_BASELINES = ["compat_lw", "lw_cos"]

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
LSIZE = {"layer1": _SIZES[0] + _SIZES[1], "layer2": _SIZES[2] + _SIZES[3], "head": _SIZES[4] + _SIZES[5]}

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
n_train = {cid: len(clients[cid]["y_train"]) for cid in IDS}; N_TOTAL = sum(n_train.values())
NTR = np.array([n_train[k] for k in IDS]); EYE = np.eye(nK, dtype=bool)
PREV = np.array([np.mean(clients[k]["y_train"]) for k in IDS])
SIZEQ = pd.qcut(pd.Series(NTR).rank(method="first"), 4, labels=False).values
STRAT = (NTR > np.median(NTR)).astype(int)
LOW, HIGH, ALLK = np.where(SIZEQ == 0)[0], np.where(SIZEQ == 3)[0], np.arange(nK)
if USE_TEST: assert all("X_test" in clients[k] for k in IDS), "USE_TEST=True mais X_test absent"

def raw_stats(c):
    """Signature en espace BRUT (les features standardisées par client ont moyenne 0 / écart-type 1 pour tous)."""
    if "raw_mean" in c and "raw_scale" in c: return np.asarray(c["raw_mean"], float), np.asarray(c["raw_scale"], float), "raw_*"
    sc = c.get("scaler", None)
    if sc is not None and hasattr(sc, "mean_"): return np.asarray(sc.mean_, float), np.asarray(sc.scale_, float), "scaler"
    Xn = np.asarray(c["X_train"], float); return Xn.mean(0), Xn.std(0), "STANDARDISÉ (signature quasi constante)"

data, SRC = {}, set()
for cid in IDS:
    c = clients[cid]; mu_, sd_, src = raw_stats(c); SRC.add(src)
    d = dict(Xtr_dev=T(c["X_train"], DEVICE), ytr_dev=T(c["y_train"], DEVICE), Xtr=T(c["X_train"]),
             Xva=T(c["X_val"]), yva=np.asarray(c["y_val"], dtype=float),
             sig_static=np.concatenate([mu_, np.log(sd_ + EPS), [np.mean(c["y_train"])]]))
    if USE_TEST: d["Xte"], d["yte"] = T(c["X_test"]), np.asarray(c["y_test"], dtype=float)
    data[cid] = d
print(f"level={LEVEL} {META.get('hetero_params')} | {nK} clients | n_train {NTR.min()}/{int(np.median(NTR))}/{NTR.max()} | "
      f"prev {PREV.min():.3f}/{np.median(PREV):.3f}/{PREV.max():.3f} | params={P} | éval={'TEST' if USE_TEST else 'val-EVAL'} | "
      f"signature: {SRC} | true_compat: {'oui' if TRUE is not None else 'NON (oracle_truth désactivé)'}")

def make_splits(seed):
    sel, ev = {}, {}
    for cid in IDS:
        X, y = data[cid]["Xva"], data[cid]["yva"]; n = len(y)
        perm = np.random.default_rng(seed + 7919 * cid).permutation(n)
        if USE_TEST:
            sel[cid] = (X, y); ev[cid] = (data[cid]["Xte"], data[cid]["yte"])
        else:
            ns = int(round(SEL_FRAC * n)); i1, i2 = perm[:ns], perm[ns:]
            sel[cid] = (X[torch.as_tensor(i1)], y[i1]); ev[cid] = (X[torch.as_tensor(i2)], y[i2])
    return sel, ev

def similarity(models):
    rows = []
    for k in IDS:
        p = probs_of(models[k], data[k]["Xtr"])
        rows.append(np.concatenate([data[k]["sig_static"], [p.mean(), p.std()]]))
    Z = np.vstack(rows); Z = (Z - Z.mean(0)) / (Z.std(0) + EPS)
    if BALANCE_BLOCKS: Z[:, :2 * INPUT_DIM] /= np.sqrt(2 * INPUT_DIM)
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.exp(-D2 / (np.median(D2[~EYE]) + EPS))

def topk_mask(score, allowed, k=K_NEIGHBORS):
    out = np.zeros(len(allowed), bool); idx = np.where(allowed)[0]
    if len(idx): out[idx[np.argsort(-score[idx])[:k]]] = True
    return out


def pre_counts(FTt, V, sel):
    cnt = {}
    for ki, k in enumerate(IDS):
        Xs, ys = sel[k]; ys = np.asarray(ys, float); ns = len(ys)
        Lb = (batch_probs(FTt[ki][None], Xs)[0] - ys) ** 2
        CL = ((batch_probs(FTt, Xs) - ys[None]) ** 2).mean(1); cp = CL[ki] - CL; cp[ki] = -np.inf
        valid = np.arange(nK) != ki; cand = topk_mask(cp, valid); c = {}
        for l in HIDL:
            ds = Lb[None] - (batch_probs(FTt[ki][None] + GAMMA * V[l], Xs) - ys[None]) ** 2
            g = ds.mean(1); t = np.nan_to_num(g / (ds.std(1, ddof=1) / np.sqrt(ns) + EPS), nan=-9.0)
            g[ki] = -np.inf; t[ki] = -np.inf
            c[l] = int((cand & (t > PRIMARY_TC) & (g > 0)).sum())
        cnt[ki] = c
    return cnt


# Mesure
def new_acc():
    z = lambda: np.zeros((nR, nK)); nl = len(LAYERS)
    return dict(SB=z(), ACC=z(), POS=z(), HARM=z(), COMM=z(), FTB=np.zeros(nK), M=0,
                GM=np.zeros((nl, nK, nK)), SS=np.zeros((nK, nK)), CP=np.zeros((nK, nK)),
                REL=np.zeros((nl, nK)), RELN=np.zeros((nl, nK)), SIG=np.zeros(nl), SIGN=0, NSEL=np.zeros((2, nK)))

def measure(FTt, Gprev, Gnew, sel, ev, acc, rng):
    S = similarity({k: FTt[i] for i, k in enumerate(IDS)}); Dt = FTt - Gprev[None]; acc["M"] += 1; acc["SS"] += S
    V = {l: Dt * CMASK[l] for l in LAYERS}; COS = {}
    for l in LAYERS:
        Vn = V[l] / (V[l].norm(dim=1, keepdim=True) + EPS); COS[l] = (Vn @ Vn.T).numpy()
    cnt = pre_counts(FTt, V, sel)                                         # nb de sources retenues par le gate pour chaque client (utilisé par shufS)
    prev_nsel = None                                                      # compte du client précédent (compat_count_shuffled)
    pool_prev = {l: [] for l in HIDL}                                     # historique des comptes (shuf20)
    rng_p = np.random.default_rng(1000 + acc["M"])                        # rng des permutations (shuf20 et shufS)
    for ki, k in enumerate(IDS):
        Xs, ys = sel[k]; Xe, ye = ev[k]; ns, ne = len(ys), len(ye)
        if ns < 5 or ne == 0: continue
        Xc, yc = torch.cat([Xs, Xe]), np.concatenate([ys, ye]).astype(float)
        Lb = (batch_probs(FTt[ki][None], Xc)[0] - yc) ** 2; Lb_e = Lb[ns:]; acc["FTB"][ki] += Lb_e.mean()
        valid = np.arange(nK) != ki
        CL = ((batch_probs(FTt, Xs) - np.asarray(ys, float)[None]) ** 2).mean(1); cp = CL[ki] - CL   # compat : >0 = j prédit mieux que k
        acc["CP"][ki] += np.where(valid, cp, 0.0); cp[ki] = -np.inf
        gS, gE, tS = {}, {}, {}
        for li, l in enumerate(LAYERS):
            Pm = batch_probs(FTt[ki][None] + GAMMA * V[l], Xc)
            Dd = Lb[None] - (Pm - yc[None]) ** 2; ds = Dd[:, :ns]
            gS[l] = ds.mean(1); tS[l] = np.nan_to_num(gS[l] / (ds.std(1, ddof=1) / np.sqrt(ns) + EPS), nan=-9.0)
            gE[l] = Dd[:, ns:].mean(1)
            acc["SIG"][li] += ds[valid].std(1, ddof=1).mean()
            r_ = spearmanr(gS[l][valid], gE[l][valid]).correlation       # est-ce que le gain sur SEL prédit le gain sur EVAL ?
            if np.isfinite(r_): acc["REL"][li, ki] += r_; acc["RELN"][li, ki] += 1
            acc["GM"][li, ki] += np.where(valid, gE[l], 0.0)
            gS[l][ki] = -np.inf; tS[l][ki] = -np.inf; gE[l][ki] = np.nan
        acc["SIGN"] += 1
        cand_sim = topk_mask(S[ki], valid); cand_cp = topk_mask(cp, valid)
        gate = lambda l, t: (tS[l] > t) & (gS[l] > 0)
        plans = {"uniform": {l: valid for l in LAYERS}, "sim": {l: cand_sim for l in LAYERS},
                 "compat": {l: cand_cp for l in LAYERS}, "compat_lw": {l: cand_cp for l in HIDL},
                 "lw_cos": {l: topk_mask(COS[l][ki], valid & (COS[l][ki] > 0)) for l in LAYERS}}
        for t in TC_GRID: plans[f"sel@{t}"] = {l: cand_cp & gate(l, t) for l in HIDL}
        plans[f"sel_sim@{PRIMARY_TC}"] = {l: cand_sim & gate(l, PRIMARY_TC) for l in HIDL}
        plans[f"sel_all@{PRIMARY_TC}"] = {l: cand_cp & gate(l, PRIMARY_TC) for l in LAYERS}
        rm = {}                                                           # contrôle : mêmes candidats et même nombre de sources que sel, mais tirées au hasard
        for l in HIDL:
            n_ = int(plans[PROPOSED][l].sum()); idx = np.where(cand_cp)[0]; m_ = np.zeros(nK, bool)
            if n_ > 0 and len(idx): m_[rng.choice(idx, min(n_, len(idx)), replace=False)] = True
            rm[l] = m_
        plans["compat_rand_matched"] = rm

        

        # contrôles sans gate : top-1, top-2, et top-n avec le compte du client précédent
        nsel = {l: int(plans[PROPOSED][l].sum()) for l in HIDL}
        ns_sh = nsel if prev_nsel is None else prev_nsel
        kk = lambda n: topk_mask(cp, valid, k=n) if n > 0 else np.zeros(nK, bool)
        plans["compat_k1"] = {l: kk(1) for l in HIDL}
        plans["compat_k2"] = {l: kk(2) for l in HIDL}
        plans["compat_count_shuffled"] = {l: kk(ns_sh[l]) for l in HIDL}
        prev_nsel = nsel

        # abstention selon la taille, gate utilisé seulement pour décider "si", et 20 permutations du compte
        small = n_train[k] <= np.median(NTR)
        zero = np.zeros(nK, bool)
        plans["compat_k2_small"] = {l: (kk(2) if small else zero) for l in HIDL}
        anyg = {l: bool(plans[PROPOSED][l].any()) for l in HIDL}
        plans["compat_k2_gatewhether"] = {l: (kk(2) if anyg[l] else zero) for l in HIDL}
        pool = {l: (pool_prev[l] if pool_prev[l] else [nsel[l]]) for l in HIDL}
        for j in range(N_PERM):
            plans[f"shuf20_{j}"] = {l: kk(int(rng_p.choice(pool[l]))) for l in HIDL}
        for l in HIDL: pool_prev[l].append(nsel[l])

        # comptes permutés à l'intérieur de la même strate de taille
        same_s = [j for j in range(nK) if j != ki and STRAT[j] == STRAT[ki]]
        for r_ in range(N_SHUFS):
            j = same_s[rng_p.integers(len(same_s))]
            plans[f"shufS_{r_}"] = {l: (topk_mask(cp, valid, k=cnt[j][l]) if cnt[j][l] > 0 else zero) for l in HIDL}
        for li_, l in enumerate(HIDL): acc["NSEL"][li_, ki] += nsel[l]    # pour Spearman(nsel, n_train)
        # fin des contrôles

        if TRUE is not None:
            ct = topk_mask(TRUE[ki], valid)
            plans["oracle_truth"] = {l: ct for l in HIDL}; plans["oracle_truth_gate"] = {l: ct & gate(l, PRIMARY_TC) for l in HIDL}
        selfv = FTt[ki] + GAMMA * sum(V[l][ki] for l in HIDL)
        names, vecs = ["fedavg_global", "self_extrap"], [Gnew, selfv]
        for n, pl in plans.items():
            v = FTt[ki].clone()
            for l, m in pl.items():
                if m.any(): v = v + GAMMA * (torch.tensor(m / m.sum(), dtype=torch.float32) @ V[l])
            names.append(n); vecs.append(v)
        Pr = batch_probs(torch.stack(vecs), Xe)
        for i, n in enumerate(names):
            ri = RIDX[n]; acc["SB"][ri, ki] += Lb_e.mean() - ((Pr[i] - ye) ** 2).mean()
            if n in NOPAIR: continue
            for l, m in plans[n].items():
                g = gE[l][m]; acc["ACC"][ri, ki] += m.sum(); acc["POS"][ri, ki] += (g > 0).sum()
                acc["HARM"][ri, ki] += (g < -HARM_DELTA).sum(); acc["COMM"][ri, ki] += m.sum() * LSIZE[l]



def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    G = parameters_to_vector(SimpleMLP().parameters()).detach().cpu().clone()
    sel, ev = make_splits(seed); acc, t0 = new_acc(), time.time(); rng = np.random.default_rng(seed)
    for r in range(1, N_ROUNDS + 1):
        FT = {k: train_local(G, k, seed + r * 1000 + k) for k in IDS}
        Gnew = fedavg(FT)
        if r in MEAS_ROUNDS: measure(torch.stack([FT[k] for k in IDS]), G, Gnew, sel, ev, acc, rng)
        G = Gnew
    print(f"[seed={seed}] terminé en {(time.time() - t0) / 60:.1f} min")
    return acc, int(np.median([len(sel[k][1]) for k in IDS]))

_V = torch.stack([parameters_to_vector(SimpleMLP().parameters()).detach() for _ in range(3)]); _X = data[IDS[0]]["Xva"][:64]
assert np.allclose(batch_probs(_V, _X)[1], probs_of(_V[1], _X), atol=1e-5), "batch_probs != probs_of"
t00 = time.time(); OUT = [run_seed(s) for s in SEEDS]; ACCS = [o[0] for o in OUT]; NS_MED = OUT[0][1]

# Analyse
SM = sum(a["M"] for a in ACCS); agg = lambda key: sum(a[key] for a in ACCS)
dB = agg("SB") / SM; ACC_, POS_, HARM_, COMM_ = agg("ACC"), agg("POS"), agg("HARM"), agg("COMM")
GMm = agg("GM") / SM; SSm = agg("SS") / SM; CPm = agg("CP") / SM; FTB = agg("FTB") / SM
RELm = agg("REL") / np.maximum(agg("RELN"), 1); SIGm = agg("SIG") / max(agg("SIGN"), 1) if np.isscalar(agg("SIGN")) else None
SIGm = agg("SIG") / agg("SIGN")
for li in range(len(LAYERS)): GMm[li][EYE] = np.nan
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 70)

def ci(v, B=BOOT_B):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 5: return (np.nan,) * 3
    bs = v[np.random.default_rng(1).integers(0, len(v), (B, len(v)))].mean(1)
    return v.mean(), *np.percentile(bs, [2.5, 97.5])
def fm(t, s=1e3, nd=3): return "n/a" if np.isnan(t[0]) else f"{t[0]*s:+.{nd}f} [{t[1]*s:+.{nd}f}, {t[2]*s:+.{nd}f}]"
def seedv(r): return [float(a["SB"][RIDX[r]].mean() / a["M"]) for a in ACCS]
def seed_contrast(a_, b_): return [float((a["SB"][RIDX[a_]] - a["SB"][RIDX[b_]]).mean() / a["M"]) for a in ACCS]
def pairstat(r, M_):
    a = ACC_[RIDX[r]].sum(); return M_[RIDX[r]].sum() / a if a > 0 else np.nan

def table(rules):
    rows = []
    for r in rules:
        i = RIDX[r]
        rows.append(dict(rule=r, comm_Mparams=round(COMM_[i].mean() / SM / 1e6, 4), pairs_per_client=round(ACC_[i].mean() / SM, 1),
                         prec_pair=round(pairstat(r, POS_), 3), harm_pair=round(pairstat(r, HARM_), 3),
                         dB_all_x1e3=fm(ci(dB[i])), rel_dBrier=f"{dB[i].mean() / FTB.mean():+.4%}",
                         dB_LOW=fm(ci(dB[i][LOW])), dB_HIGH=fm(ci(dB[i][HIGH])),
                         P10=f"{np.percentile(dB[i], 10)*1e3:+.3f}", per_seed="/".join(f"{x*1e3:+.2f}" for x in seedv(r))))
    return pd.DataFrame(rows)

print("=" * 130 + f"\nA. RÉSULTATS [{LEVEL}] seeds={SEEDS} | FT Brier moyen={FTB.mean():.4f} | dBrier x1e3 (>0 = mieux que FT_k)\n" + "=" * 130)
print(table(RULES).to_string(index=False))

print("\n" + "=" * 130 + "\nB. CONTRASTES APPARIÉS PAR CLIENT : PROPOSED - baseline (IC95 bootstrap sur les 48 clients)\n" + "=" * 130)
rows = []
for b in [r for r in RULES if r != PROPOSED]:
    d = dB[RIDX[PROPOSED]] - dB[RIDX[b]]; sc = seed_contrast(PROPOSED, b)
    rows.append(dict(vs=b, all=fm(ci(d)), LOW=fm(ci(d[LOW])), HIGH=fm(ci(d[HIGH])), seeds_pos=f"{sum(s > 0 for s in sc)}/{len(sc)}"))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 130 + "\nC. SIGNAL PAR PAIRE : le gain mesuré sur SEL prédit-il celui sur EVAL ? (le gate ne peut pas faire mieux que cela)\n" + "=" * 130)
print(f"taille médiane de SEL = {NS_MED} ; ")
rows = [dict(layer=l, sigma_gain_par_echantillon=round(float(SIGm[li]), 5), SE_par_paire_x1e3=round(float(SIGm[li] / np.sqrt(NS_MED)) * 1e3, 3),
             spearman_SEL_vs_EVAL=fm(ci(RELm[li]), 1, 3)) for li, l in enumerate(LAYERS)]
print(pd.DataFrame(rows).to_string(index=False))
print("  SE_par_paire à comparer au gain moyen par paire (colonne dB ci-dessus, de l'ordre de 1e-4) : si SE >> gain, aucun gate par paire n'est puissant.")

print("\n" + "=" * 130 + "\nD. LA COMPATIBILITÉ ESTIMÉE RETROUVE-T-ELLE LA VÉRITÉ ET LE GAIN RÉEL ?\n" + "=" * 130)
def rowcorr(score, ref, k):
    v = np.arange(nK) != k; return spearmanr(score[k][v], ref[k][v]).correlation
rows = []
if TRUE is not None:
    rs, rc = [rowcorr(SSm, TRUE, k) for k in range(nK)], [rowcorr(CPm, TRUE, k) for k in range(nK)]
    rows.append(dict(ref="compat vraie", sim=fm(ci(rs), 1, 3), compat=fm(ci(rc), 1, 3), compat_moins_sim=fm(ci(np.array(rc) - np.array(rs)), 1, 3)))
for li, l in enumerate(LAYERS):
    rs, rc = [rowcorr(SSm, GMm[li], k) for k in range(nK)], [rowcorr(CPm, GMm[li], k) for k in range(nK)]
    top = lambda sc: [np.nanmean(GMm[li][k][np.where(np.arange(nK) != k)[0][np.argsort(-sc[k][np.where(np.arange(nK) != k)[0]])[:K_NEIGHBORS]]]) for k in range(nK)]
    rows.append(dict(ref=f"gain réel {l}", sim=fm(ci(rs), 1, 3), compat=fm(ci(rc), 1, 3), compat_moins_sim=fm(ci(np.array(rc) - np.array(rs)), 1, 3),
                     top8_gain_sim_x1e3=fm(ci(top(SSm))), top8_gain_compat_x1e3=fm(ci(top(CPm))),
                     toutes_sources_x1e3=fm(ci([np.nanmean(GMm[li][k]) for k in range(nK)]))))
print(pd.DataFrame(rows).to_string(index=False))
print("  Rappel : 'compat' est un score client x client (Brier de la source sur la cible) ; la couche n'intervient que dans le gate.")

print("\n" + "=" * 130 + f"\nE. VERDICT (critères fixés avant le run) — niveau {LEVEL}\n" + "=" * 130)
ip = RIDX[PROPOSED]; m_, lo_, hi_ = ci(dB[ip]); chk = []
def crit(name, ok, info): chk.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ->  {info}")
if LEVEL == "iid":
    bad = [r for r in RULES if r not in NOPAIR + ("fedavg_global",) and ci(dB[RIDX[r]])[1] > 0]
    crit("N0 contrôle IID : aucune règle n'a un IC > 0", len(bad) == 0, f"règles avec IC > 0 : {bad if bad else 'aucune'}")
    print("  (N0 est le seul critère pertinent sur ce niveau. Si N0 échoue, le pipeline fabrique du gain : arrêter.)")
else:
    if TRUE is not None:
        o = ci(dB[RIDX["oracle_truth"]]); crit("N1 oracle_truth - FT > 0 (IC95) : il existe du headroom", o[1] > 0, fm(o))
        rs, rc = np.array([rowcorr(SSm, TRUE, k) for k in range(nK)]), np.array([rowcorr(CPm, TRUE, k) for k in range(nK)])
        d2 = ci(rc - rs); crit("N2 compat retrouve la compatibilité vraie, mieux que sim", ci(rc)[1] > 0 and d2[1] > 0, f"compat {fm(ci(rc),1,3)} | diff {fm(d2,1,3)}")
    else: print("  N1/N2 non évaluables : pas de true_compat dans la métadonnée.")
    best_lw = max(LW_BASELINES, key=lambda r: dB[RIDX[r]].mean())
    d3 = ci(dB[ip] - dB[RIDX[best_lw]]); crit(f"N3 {PROPOSED} - {best_lw} > 0 (IC95) : le gate apporte quelque chose", d3[1] > 0, fm(d3))
    d3b = ci(dB[ip] - dB[RIDX["compat_rand_matched"]]); crit("N3b vs mêmes candidats, même nombre de sources, au hasard (IC95 > 0)", d3b[1] > 0, fm(d3b))
    crit(f"N4 {PROPOSED} - FT > 0 (IC95) et > 0 sur chaque seed", lo_ > 0 and all(s > 0 for s in seedv(PROPOSED)), f"{fm((m_, lo_, hi_))} | seeds " + "/".join(f"{s*1e3:+.2f}" for s in seedv(PROPOSED)))
    d5 = ci(dB[ip] - dB[RIDX["self_extrap"]]); crit("N5 vs self_extrap (plus d'adaptation locale) > 0 (IC95)", d5[1] > 0, fm(d5))
    print(f"\n  Gain relatif de {PROPOSED} : {m_ / FTB.mean():+.4%} du Brier de FT.   fedavg_global : {dB[RIDX['fedavg_global']].mean() / FTB.mean():+.1%}")
    print(f"  => {'SIGNAL CONFIRMÉ' if all(chk) else 'NON CONFIRMÉ : ' + str(sum(not c for c in chk)) + ' critère(s) en échec sur ' + str(len(chk))}")

cdf = pd.DataFrame(dict(client=IDS, n_train=NTR, prev=PREV, FT_brier=FTB, size_quartile=SIZEQ,
                        **{f"dB_x1e3[{r}]": dB[RIDX[r]] * 1e3 for r in RULES}))
cdf.to_csv(os.path.join(RESULTS_DIR, f"client_level_{LEVEL}.csv"), index=False)
print(f"\nCSV par client : {RESULTS_DIR}/client_level_{LEVEL}.csv | Total time: {(time.time() - t00) / 60:.1f} min")