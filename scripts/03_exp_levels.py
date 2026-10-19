# CLST-FL : EXP-Levels. Mêmes règles et même config que EXP-Final, seul LEVEL change.
# iid sert de contrôle N0 ; low et medium montrent comment l'effet évolue avec l'hétérogénéité.
# Critères fixés avant le run (section D). N'écrit aucun fichier, tout est affiché.
import os, json, pickle, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.stats import spearmanr
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config (identique à EXP-Final)
BASE_DIR = "data_v3"   # dossier écrit par data_00_build_clients_v3.py ; à adapter
LEVELS = None                            # None : tous les niveaux sauf "high" ; sinon par ex. ["iid", "medium"]
SEEDS = [223, 224, 225, 226, 227]        # ~9 min par seed et par niveau ; passer à 3 seeds si c'est trop long
N_ROUNDS, WARMUP, MEAS_EVERY = 50, 3, 5
MEAS_ROUNDS = list(range(WARMUP + MEAS_EVERY, N_ROUNDS + 1, MEAS_EVERY))
LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE = 5, 1e-3, 64
HIDDEN, DROPOUT = (64, 32), 0.10
K, GAMMA = 2, 0.10
TC_P, TC_S = 1.645, 2.0
HARM_DELTA, BOOT_B, CVAR_Q = 1e-4, 2000, 0.10
NI_MARGIN = 0.2e-3                       # même valeur que EXP-Final pour pouvoir comparer ; l'IC est plus informatif
EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS = ["layer1", "layer2", "head"]; HIDL = ["layer1", "layer2"]
PROPOSED, STRICT = f"sel@{TC_P}", f"sel@{TC_S}"

avail = sorted(d[len("clients_hetero_"):] for d in os.listdir(BASE_DIR)
               if d.startswith("clients_hetero_") and os.path.isdir(os.path.join(BASE_DIR, d)))
print("Niveaux disponibles :", avail)
if LEVELS is None: LEVELS = [l for l in avail if l != "high"]
LEVELS = sorted(LEVELS, key=lambda l: l != "iid")
assert LEVELS and all(l in avail for l in LEVELS), f"LEVELS invalides : {LEVELS} (disponibles : {avail})"
print(f"Niveaux lancés : {LEVELS} | seeds={SEEDS} | durée estimée ≈ {len(LEVELS) * len(SEEDS) * 9 / 60:.1f} h")

def load_meta(level):
    with open(os.path.join(BASE_DIR, f"clients_hetero_{level}", "metadata.json"), "r") as f: return json.load(f)
INPUT_DIM = load_meta(LEVELS[0])["input_dim"]

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

# Données (rechargées par niveau)
def T(x, dev="cpu"): return torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)

def setup_level(level):
    global LEVEL, META, TRUE, RULES, RIDX, nR, IDS, nK, n_train, N_TOTAL, NTR, PREV, SIZEQ, SMALL, data
    LEVEL = level; cdir = os.path.join(BASE_DIR, f"clients_hetero_{level}")
    META = load_meta(level); assert META["input_dim"] == INPUT_DIM, "input_dim différent entre niveaux"
    TRUE = np.array(META["true_compat"]) if "true_compat" in META else None
    RULES = ["compat_k2", "compat_k2_small", "compat_count_shuffled", "rand_matched_all", PROPOSED, STRICT]
    if TRUE is not None: RULES.append("oracle_truth_gate")
    RIDX = {n: i for i, n in enumerate(RULES)}; nR = len(RULES)
    clients = {}
    for fname in sorted(os.listdir(cdir)):
        if fname.startswith("client_") and fname.endswith(".pkl"):
            with open(os.path.join(cdir, fname), "rb") as f: c = pickle.load(f)
            clients[c["client_id"]] = c
    IDS = sorted(clients); nK = len(IDS)
    assert all("X_test" in clients[k] for k in IDS), "X_test absent"
    n_train = {cid: len(clients[cid]["y_train"]) for cid in IDS}; N_TOTAL = sum(n_train.values())
    NTR = np.array([n_train[k] for k in IDS]); PREV = np.array([np.mean(clients[k]["y_train"]) for k in IDS])
    SIZEQ = pd.qcut(pd.Series(NTR).rank(method="first"), 4, labels=False).values
    SMALL = NTR <= np.median(NTR)
    data = {}
    for cid in IDS:
        c = clients[cid]
        data[cid] = dict(Xtr_dev=T(c["X_train"], DEVICE), ytr_dev=T(c["y_train"], DEVICE),
                         Xva=T(c["X_val"]), yva=np.asarray(c["y_val"], dtype=float),
                         Xte=T(c["X_test"]), yte=np.asarray(c["y_test"], dtype=float))
    print(f"\n{'#' * 140}\nlevel={LEVEL} {META.get('hetero_params')} | {nK} clients | n_train {NTR.min()}/{int(np.median(NTR))}/{NTR.max()} | "
          f"prev {PREV.min():.3f}/{np.median(PREV):.3f}/{PREV.max():.3f} | true_compat: {'oui' if TRUE is not None else 'NON'} | règles: {RULES}\n{'#' * 140}")

def make_splits():
    return ({cid: (data[cid]["Xva"], data[cid]["yva"]) for cid in IDS},      # sélection = val complète
            {cid: (data[cid]["Xte"], data[cid]["yte"]) for cid in IDS})      # évaluation = test

def topk_mask(score, allowed, k=K):
    out = np.zeros(len(allowed), bool); idx = np.where(allowed)[0]
    if len(idx) and k > 0: out[idx[np.argsort(-score[idx])[:k]]] = True
    return out

# Mesure
def new_acc():
    z = lambda: np.zeros((nR, nK))
    return dict(SB=z(), ACC=z(), POS=z(), HARM=z(), NSRC=z(), COV1=z(), COV2=z(), COMM=z(), FTB=np.zeros(nK), M=0)

def measure(FTt, Gprev, sel, ev, acc, seed):
    acc["M"] += 1
    Dt = FTt - Gprev[None]
    V = {l: Dt * CMASK[l] for l in HIDL}
    info = {}
    for ki, k in enumerate(IDS):
        Xs, ys = sel[k]; Xe, ye = ev[k]; ns, ne = len(ys), len(ye)
        if ns < 5 or ne == 0: continue
        ys = np.asarray(ys, float); ye = np.asarray(ye, float)
        Xc, yc = torch.cat([Xs, Xe]), np.concatenate([ys, ye])
        Lb = (batch_probs(FTt[ki][None], Xc)[0] - yc) ** 2
        valid = np.arange(nK) != ki
        CL = ((batch_probs(FTt, Xs) - ys[None]) ** 2).mean(1)
        cp = CL[ki] - CL; cp[ki] = -np.inf
        cand = topk_mask(cp, valid, K)
        gE, gate = {}, {}
        for l in HIDL:
            Pm = batch_probs(FTt[ki][None] + GAMMA * V[l], Xc)
            Dd = Lb[None] - (Pm - yc[None]) ** 2; ds = Dd[:, :ns]
            g = ds.mean(1); t = np.nan_to_num(g / (ds.std(1, ddof=1) / np.sqrt(ns) + EPS), nan=-9.0)
            ge = Dd[:, ns:].mean(1); ge[ki] = np.nan; gE[l] = ge
            for tc in (TC_P, TC_S): gate[(l, tc)] = valid & (t > tc) & (g > 0)
        info[ki] = dict(cp=cp, cand=cand, valid=valid, gE=gE, gate=gate, Xe=Xe, ye=ye, Lb_e=float(Lb[ns:].mean()))
    rng = np.random.default_rng(10_000 * seed + acc["M"])
    cnt = np.zeros((nK, len(HIDL)), int)
    for ki, d in info.items():
        for li, l in enumerate(HIDL): cnt[ki, li] = int((d["cand"] & d["gate"][(l, TC_P)]).sum())
    perm = rng.permutation(nK); zero = np.zeros(nK, bool)
    for ki, d in info.items():
        cand, cp, valid, gate = d["cand"], d["cp"], d["valid"], d["gate"]
        ra = {}
        for li, l in enumerate(HIDL):
            m_ = np.zeros(nK, bool); n_ = int(cnt[ki, li]); idx = np.where(valid)[0]
            if n_ > 0: m_[rng.choice(idx, min(n_, len(idx)), replace=False)] = True
            ra[l] = m_
        plans = {
            "compat_k2": {l: cand for l in HIDL},
            "compat_k2_small": {l: (cand if SMALL[ki] else zero) for l in HIDL},
            "compat_count_shuffled": {l: topk_mask(cp, valid, k=int(cnt[perm[ki], li])) for li, l in enumerate(HIDL)},
            "rand_matched_all": ra,
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
                acc["COMM"][ri, ki] += m.sum() * LSIZE[l]
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
    print(f"[{LEVEL} | seed={seed}] terminé en {(time.time() - t0) / 60:.1f} min")
    return acc

# Outils stats
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
def cvar(v, q=CVAR_Q):
    v = np.sort(np.asarray(v, float)); m = max(1, int(np.ceil(q * len(v)))); return v[:m].mean()
def boot(fn, pool=None, B=BOOT_B, seed=1):
    pool = np.arange(nK) if pool is None else np.asarray(pool)
    rg = np.random.default_rng(seed)
    bs = np.array([fn(pool[rg.integers(0, len(pool), len(pool))]) for _ in range(B)])
    return fn(pool), *np.percentile(bs, [2.5, 97.5])
def fm(t, s=1e3, nd=3): return f"{t[0]*s:+.{nd}f} [{t[1]*s:+.{nd}f}, {t[2]*s:+.{nd}f}]"

# Analyse d'un niveau
def analyze(ACCS):
    SM = sum(a["M"] for a in ACCS); agg = lambda key: sum(a[key] for a in ACCS)
    dB = agg("SB") / SM; FTB = agg("FTB") / SM
    ACC_, POS_, HARM_, COMM_ = agg("ACC"), agg("POS"), agg("HARM"), agg("COMM")
    NSRC, COV1, COV2 = agg("NSRC") / SM, agg("COV1") / SM, agg("COV2") / SM
    HF = np.where(ACC_ > 0, HARM_ / np.maximum(ACC_, 1), 0.0)
    HE = HARM_ / (SM * len(HIDL) * K)
    seed_HF = lambda a, ri: float(np.where(a["ACC"][ri] > 0, a["HARM"][ri] / np.maximum(a["ACC"][ri], 1), 0.0).mean())
    ip = RIDX[PROPOSED]

    print("=" * 140 + f"\nA1. PERFORMANCE PAR CLIENT [{LEVEL}] seeds={SEEDS} K={K} | FT Brier moyen={FTB.mean():.4f} | dBrier x1e3 (>0 = mieux que FT_k)\n" + "=" * 140)
    rows = []
    for r in RULES:
        v = dB[RIDX[r]]; ps = [float((a["SB"][RIDX[r]] / a["M"]).mean()) for a in ACCS]
        rows.append(dict(rule=r, dB_mean=fm(boot(lambda i: v[i].mean())), median=f"{np.median(v)*1e3:+.3f}", P10=f"{np.percentile(v, 10)*1e3:+.3f}",
                         CVaR10=fm(boot(lambda i: cvar(v[i]))), worst=f"{v.min()*1e3:+.3f}", per_seed="/".join(f"{x*1e3:+.2f}" for x in ps)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 140 + f"\nA2. SÉLECTION, SÉCURITÉ, COMMUNICATION [{LEVEL}]\n" + "=" * 140)
    rows = []
    for r in RULES:
        i = RIDX[r]; a = ACC_[i].sum()
        rows.append(dict(rule=r, n_src_mean=round(NSRC[i].mean(), 3), cov_ge1=round(COV1[i].mean(), 3), cov_ge2=round(COV2[i].mean(), 3),
                         precision_pair=round(POS_[i].sum() / a, 3) if a > 0 else np.nan, harm_pair=round(HARM_[i].sum() / a, 3) if a > 0 else np.nan,
                         harm_frac_client=fm(boot(lambda j: HF[i][j].mean()), 1, 3), harm_exposure=fm(boot(lambda j: HE[i][j].mean()), 1, 3),
                         comm_Mparams=round(COMM_[i].mean() / SM / 1e6, 4)))
    print(pd.DataFrame(rows).to_string(index=False))

    def contrast(a, b, pool=None):
        ia, ib = RIDX[a], RIDX[b]; d = dB[ia] - dB[ib]
        return dict(mean=boot(lambda i: d[i].mean(), pool), dcvar=boot(lambda i: cvar(dB[ia][i]) - cvar(dB[ib][i]), pool),
                    harm=boot(lambda i: (HF[ia] - HF[ib])[i].mean(), pool), hexp=boot(lambda i: (HE[ia] - HE[ib])[i].mean(), pool),
                    seeds_mean=sum(float(((x["SB"][ia] - x["SB"][ib]).mean() / x["M"])) > 0 for x in ACCS),
                    seeds_harm=sum(seed_HF(x, ia) - seed_HF(x, ib) < 0 for x in ACCS))

    print("\n" + "=" * 140 + f"\nB. CONTRASTES APPARIÉS [{LEVEL}] (IC95 bootstrap sur les clients) — mean/dCVaR10 en dBrier x1e3 ; harm en fraction\n" + "=" * 140)
    PAIRS = [(PROPOSED, "compat_k2"), (PROPOSED, "compat_count_shuffled"), (PROPOSED, "rand_matched_all"), (PROPOSED, "compat_k2_small"),
             (STRICT, "compat_k2"), (PROPOSED, STRICT)]
    if TRUE is not None: PAIRS.append(("oracle_truth_gate", PROPOSED))
    rows = []
    for a, b in PAIRS:
        c = contrast(a, b)
        rows.append(dict(contraste=f"{a} - {b}", mean=fm(c["mean"]), dCVaR10=fm(c["dcvar"]), harm_frac=fm(c["harm"], 1, 3), harm_expo=fm(c["hexp"], 1, 3),
                         seeds_mean_pos=f"{c['seeds_mean']}/{len(SEEDS)}", seeds_harm_neg=f"{c['seeds_harm']}/{len(SEEDS)}"))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 140 + f"\nC. STRATIFICATION PAR TAILLE [{LEVEL}] (Q1 petit ... Q4 grand) — contraste {PROPOSED} vs compat_k2 / compat_k2_small, IC95 intra-tier\n" + "=" * 140)
    rows = []
    for q in range(4):
        pool = np.where(SIZEQ == q)[0]; c = contrast(PROPOSED, "compat_k2", pool); c2 = contrast(PROPOSED, "compat_k2_small", pool)
        rows.append(dict(tier=f"Q{q+1}", cov_sel=round(COV1[ip][pool].mean(), 3), mean_vs_k2=fm(c["mean"]), harm_vs_k2=fm(c["harm"], 1, 3),
                         mean_vs_small=fm(c2["mean"]), harm_vs_small=fm(c2["harm"], 1, 3)))
    print(pd.DataFrame(rows).to_string(index=False))
    rs = [spearmanr(dB[ip] - dB[RIDX["compat_k2"]], np.log(NTR)), spearmanr(HF[ip] - HF[RIDX["compat_k2"]], np.log(NTR))]
    print(f"\nSpearman vs log(n_train), {PROPOSED} - compat_k2 : dBrier {rs[0].correlation:+.3f} (p={rs[0].pvalue:.3f}) | dHarm {rs[1].correlation:+.3f} (p={rs[1].pvalue:.3f})  [exploratoire]")

    # D. verdict (critères fixés avant le run)
    print("\n" + "=" * 140 + f"\nD. VERDICT [{LEVEL}]\n" + "=" * 140)
    c1 = contrast(PROPOSED, "compat_k2"); c2 = contrast(PROPOSED, "compat_count_shuffled"); c3 = contrast(PROPOSED, "rand_matched_all"); c4 = contrast(PROPOSED, "compat_k2_small")
    d_ft = boot(lambda i: dB[ip][i].mean())
    a_k2, a_se = ACC_[RIDX["compat_k2"]].sum(), ACC_[ip].sum()
    hp_k2 = HARM_[RIDX["compat_k2"]].sum() / a_k2 if a_k2 > 0 else np.nan; hp_se = HARM_[ip].sum() / a_se if a_se > 0 else np.nan
    ck = []
    def crit(name, ok, info, primary=False): ck.append((ok, primary)); print(f"  [{'PASS' if ok else 'FAIL'}]{' (PRIMAIRE)' if primary else ''} {name}  ->  {info}")
    if LEVEL == "iid":
        crit(f"N0a  {PROPOSED} ne dégrade pas FT : borne basse IC95 de (sel - FT) > -{NI_MARGIN*1e3:.2f}e-3", d_ft[1] > -NI_MARGIN, fm(d_ft), primary=True)
        crit("N0b  harm par paire du gate <= harm par paire de compat_k2 (estimations ponctuelles)", bool(np.isfinite(hp_se) and np.isfinite(hp_k2) and hp_se <= hp_k2),
             f"sel {hp_se:.3f} | compat_k2 {hp_k2:.3f}")
        bad = [r for r in RULES if ci_lo(dB[RIDX[r]]) > 0] if False else [r for r in RULES if boot(lambda i, r=r: dB[RIDX[r]][i].mean())[1] > 0]
        print(f"\n  [INFO] règles avec gain > 0 (IC95) vs FT en IID : {bad if bad else 'aucune'}")
        print("  [INFO] En IID les sources sont interchangeables : un gain peut être réel (mesuré sur le test), il n'est pas forcément fabriqué.")
        print(f"  [INFO] coverage_ge1 du gate = {COV1[ip].mean():.3f} | mean(sel - compat_k2) = {fm(c1['mean'])} (coût d'abstention éventuel) | harm_frac vs compat_k2 = {fm(c1['harm'],1,3)}")
        print(f"  [INFO] vs compat_k2_small : mean {fm(c4['mean'])} | harm {fm(c4['harm'],1,3)}")
        n_ok = sum(ok for ok, _ in ck)
        print(f"\n  => {'N0 PASSÉ : le gate ne dégrade pas FT et ne transfère pas plus nuisible que compat_k2 sans hétérogénéité.' if n_ok == len(ck) else 'N0 ÉCHOUÉ : examiner avant de passer à la rédaction (le gate dégrade FT ou est plus nuisible que compat_k2).'}")
    else:
        crit("P1  harm_frac(sel) - harm_frac(compat_k2) < 0 (IC95 < 0) et < 0 sur >= 4/5 seeds", c1["harm"][2] < 0 and c1["seeds_harm"] >= 4, f"{fm(c1['harm'],1,3)} | seeds {c1['seeds_harm']}/{len(SEEDS)}", primary=True)
        crit("P2  dCVaR10(sel - compat_k2) > 0 (IC95 > 0)", c1["dcvar"][1] > 0, fm(c1["dcvar"]))
        crit(f"P3  mean(sel - compat_k2) non-inférieur : borne basse IC95 > -{NI_MARGIN*1e3:.2f}e-3", c1["mean"][1] > -NI_MARGIN, fm(c1["mean"]))
        crit("P4  vs compat_count_shuffled : mean > 0 (IC95) OU harm < 0 (IC95)", c2["mean"][1] > 0 or c2["harm"][2] < 0, f"mean {fm(c2['mean'])} | harm {fm(c2['harm'],1,3)}")
        crit("P5  vs rand_matched_all : mean > 0 (IC95)", c3["mean"][1] > 0, fm(c3["mean"]))
        print(f"\n  [INFO] vs compat_k2_small : mean {fm(c4['mean'])} | dCVaR10 {fm(c4['dcvar'])} | harm {fm(c4['harm'],1,3)}")
        print(f"  [INFO] Gain {PROPOSED} - FT : {fm(d_ft)}")
        prim_ok = all(ok for ok, p in ck if p); n_ok = sum(ok for ok, _ in ck)
        print("  => " + ("CONFIRMÉ à ce niveau." if prim_ok and n_ok >= 4 else "PARTIEL : primaire OK, secondaires en échec." if prim_ok else "NON CONFIRMÉ à ce niveau (primaire en échec)."))
    return dict(level=LEVEL, cov=COV1[ip].mean(), hp_k2=hp_k2, hp_se=hp_se, harm=c1["harm"], mean_ft=d_ft, mean_k2=c1["mean"], dcvar=c1["dcvar"], mean_small=c4["mean"], harm_small=c4["harm"])

# Boucle sur les niveaux
t00 = time.time(); SUMMARY = []
for lev in LEVELS:
    setup_level(lev)
    _V = torch.stack([parameters_to_vector(SimpleMLP().parameters()).detach() for _ in range(3)]); _X = data[IDS[0]]["Xva"][:64]
    assert np.allclose(batch_probs(_V, _X)[1], probs_of(_V[1], _X), atol=1e-5), "batch_probs != probs_of"
    ACCS = [run_seed(s) for s in SEEDS]
    SUMMARY.append(analyze(ACCS))
    print(f"\n[{lev}] cumul : {(time.time() - t00) / 60:.1f} min")

# E. synthèse entre niveaux
print("\n" + "=" * 140 + f"\nE. SYNTHÈSE ENTRE NIVEAUX — {PROPOSED} ; contrastes en dBrier x1e3 ; harm en fraction (a - b, <0 = gate moins nuisible)\n" + "=" * 140)
rows = [dict(level=s["level"], cov_sel=round(s["cov"], 3), harm_pair_k2=round(s["hp_k2"], 3), harm_pair_sel=round(s["hp_se"], 3),
             harm_frac_vs_k2=fm(s["harm"], 1, 3), mean_vs_FT=fm(s["mean_ft"]), mean_vs_k2=fm(s["mean_k2"]), dCVaR10_vs_k2=fm(s["dcvar"]),
             mean_vs_small=fm(s["mean_small"]), harm_vs_small=fm(s["harm_small"], 1, 3)) for s in SUMMARY]
rows.append(dict(level="high (réf. EXP-Final)", cov_sel=0.300, harm_pair_k2=0.260, harm_pair_sel=0.056, harm_frac_vs_k2="-0.196 [-0.269, -0.131]",
                 mean_vs_FT="+0.158 [+0.071, +0.268]", mean_vs_k2="+0.007 [-0.130, +0.143]", dCVaR10_vs_k2="+0.931 [+0.504, +1.285]",
                 mean_vs_small="-0.056 [-0.156, +0.040]", harm_vs_small="-0.026 [-0.077, +0.022]"))
print(pd.DataFrame(rows).to_string(index=False))
print("\nLecture : la ligne 'high' est recopiée du run précédent (mêmes seeds 223-227, mêmes données du niveau high), pas recalculée ici.")
print(f"Total time: {(time.time() - t00) / 60:.1f} min")
