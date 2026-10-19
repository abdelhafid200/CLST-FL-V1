# CLST-FL : EXP-Ablation. Gate sur le t normalisé comparé au gain brut, à la compatibilité et au gain historique, à budget égal.
# K=2 candidats (top-2 par compat) pour toutes les règles ; nouvelles seeds ; aucun fichier écrit.
# NB : le gate est un simple t par round, ce n'est pas un e-process séquentiel valide.
# Critères fixés avant le run (section E).
import os, json, pickle, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config
LEVEL = "high"                           # on peut relancer avec "medium" pour vérifier que ça tient
BASE_DIR = "data_v3"   # dossier écrit par data_00_build_clients_v3.py ; à adapter
CLIENT_DIR = os.path.join(BASE_DIR, f"clients_hetero_{LEVEL}")
SEEDS = [323, 324, 325, 326, 327]        # pas encore utilisées (42-44, 123-127 et 223-227 l'ont été)
N_ROUNDS, WARMUP, MEAS_EVERY = 50, 3, 5
MEAS_ROUNDS = list(range(WARMUP + MEAS_EVERY, N_ROUNDS + 1, MEAS_EVERY)); nM = len(MEAS_ROUNDS)
LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE = 5, 1e-3, 64
HIDDEN, DROPOUT = (64, 32), 0.10
K, GAMMA = 2, 0.10
TC_P, TC_S = 1.645, 2.0
MULTS = [0.5, 1.0, 1.5, 2.0]             # budgets = multiples du nombre de paires acceptées par sel@1.645
HARM_DELTA, BOOT_B, CVAR_Q = 1e-4, 2000, 0.10
NI_MARGIN = 0.05e-3                      # marge de non-infériorité sur la moyenne à budget égal, fixée avant le run
EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDL = ["layer1", "layer2"]; NL = len(HIDL)
PROPOSED, STRICT = f"sel@{TC_P}", f"sel@{TC_S}"
SNAMES = ["t", "g", "cp", "hg"]          # t : gain / erreur-type (le gate), g : gain brut, cp : compatibilité, hg : gain moyen historique (rounds passés + round courant)
RULE_NAMES = ["compat_k2", PROPOSED, STRICT] + [f"top_{s}@{m}x" for s in SNAMES for m in MULTS] + ["rand@1.0x", "ncount_cp", "ncount_g", "ncount_rand"]

with open(os.path.join(CLIENT_DIR, "metadata.json"), "r") as f: META = json.load(f)
INPUT_DIM = META["input_dim"]

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
LSIZE = {"layer1": _SIZES[0] + _SIZES[1], "layer2": _SIZES[2] + _SIZES[3]}

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
CMASK = dict(layer1=pmask({0, 1}), layer2=pmask({2, 3}))

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
data = {cid: dict(Xtr_dev=T(clients[cid]["X_train"], DEVICE), ytr_dev=T(clients[cid]["y_train"], DEVICE),
                  Xva=T(clients[cid]["X_val"]), yva=np.asarray(clients[cid]["y_val"], dtype=float),
                  Xte=T(clients[cid]["X_test"]), yte=np.asarray(clients[cid]["y_test"], dtype=float)) for cid in IDS}
print(f"level={LEVEL} | {nK} clients | n_train {NTR.min()}/{int(np.median(NTR))}/{NTR.max()} | K={K} | seeds={SEEDS} | "
      f"{nM} mesures/seed | NI_MARGIN={NI_MARGIN*1e3:.2f}e-3 | {len(RULE_NAMES)} règles évaluées après coup")

def make_splits():
    return ({cid: (data[cid]["Xva"], data[cid]["yva"]) for cid in IDS},      # sélection = val complète
            {cid: (data[cid]["Xte"], data[cid]["yte"]) for cid in IDS})      # évaluation = test

def topk_mask(score, allowed, k=K):
    out = np.zeros(len(allowed), bool); idx = np.where(allowed)[0]
    if len(idx) and k > 0: out[idx[np.argsort(-score[idx])[:k]]] = True
    return out

# Mesure (enregistre tout, les règles sont évaluées après coup)
def new_rec():
    f = lambda: np.full((nM, nK, NL, 2), np.nan)          # [mesure, client, couche, slot] ; slots triés par compatibilité décroissante
    return dict(CP=f(), G=f(), T=f(), GE=f(), HG=f(), DB=np.full((nM, nK, 4, 4), np.nan), FTB=np.full((nM, nK), np.nan),
                m=0, hs=np.zeros((NL, nK, nK)), hn=np.zeros(nK))

def measure(FTt, Gprev, sel, ev, rec):
    m = rec["m"]; Dt = FTt - Gprev[None]
    V = {l: Dt * CMASK[l] for l in HIDL}
    for ki, k in enumerate(IDS):
        Xs, ys = sel[k]; Xe, ye = ev[k]; ns, ne = len(ys), len(ye)
        if ns < 5 or ne == 0: continue
        ys = np.asarray(ys, float); ye = np.asarray(ye, float)
        Xc, yc = torch.cat([Xs, Xe]), np.concatenate([ys, ye])
        Lb = (batch_probs(FTt[ki][None], Xc)[0] - yc) ** 2
        valid = np.arange(nK) != ki
        CL = ((batch_probs(FTt, Xs) - ys[None]) ** 2).mean(1)
        cp = CL[ki] - CL; cp[ki] = -np.inf
        slots = np.where(topk_mask(cp, valid, K))[0]; slots = slots[np.argsort(-cp[slots])]
        rec["hn"][ki] += 1
        for li, l in enumerate(HIDL):
            Pm = batch_probs(FTt[ki][None] + GAMMA * V[l], Xc)
            Dd = Lb[None] - (Pm - yc[None]) ** 2; ds = Dd[:, :ns]
            g = ds.mean(1); t = np.nan_to_num(g / (ds.std(1, ddof=1) / np.sqrt(ns) + EPS), nan=-9.0)
            rec["hs"][li, ki] += g                                              # gain historique cumulé (toutes sources)
            rec["CP"][m, ki, li] = cp[slots]; rec["G"][m, ki, li] = g[slots]; rec["T"][m, ki, li] = t[slots]
            rec["GE"][m, ki, li] = Dd[:, ns:].mean(1)[slots]; rec["HG"][m, ki, li] = (rec["hs"][li, ki] / rec["hn"][ki])[slots]
        vecs = []                                                               # 16 modèles = 4 sous-ensembles de sources x 4 (couche1 x couche2)
        for m1 in range(4):
            for m2 in range(4):
                v = FTt[ki].clone()
                for l, mm in ((HIDL[0], m1), (HIDL[1], m2)):
                    ids = [int(slots[s]) for s in range(2) if (mm >> s) & 1]
                    if ids: v = v + GAMMA * V[l][ids].mean(0)
                vecs.append(v)
        Pr = batch_probs(torch.stack(vecs), Xe)
        rec["DB"][m, ki] = (Lb[ns:].mean() - ((Pr - ye[None]) ** 2).mean(1)).reshape(4, 4)
        rec["FTB"][m, ki] = Lb[ns:].mean()
    rec["m"] += 1

def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    G = parameters_to_vector(SimpleMLP().parameters()).detach().cpu().clone()
    sel, ev = make_splits(); rec, t0 = new_rec(), time.time()
    for r in range(1, N_ROUNDS + 1):
        FT = {k: train_local(G, k, seed + r * 1000 + k) for k in IDS}
        Gnew = fedavg(FT)
        if r in MEAS_ROUNDS: measure(torch.stack([FT[k] for k in IDS]), G, sel, ev, rec)
        G = Gnew
    assert rec["m"] == nM
    print(f"[seed={seed}] terminé en {(time.time() - t0) / 60:.1f} min")
    return rec

_V = torch.stack([parameters_to_vector(SimpleMLP().parameters()).detach() for _ in range(3)]); _X = data[IDS[0]]["Xva"][:64]
assert np.allclose(batch_probs(_V, _X)[1], probs_of(_V[1], _X), atol=1e-5), "batch_probs != probs_of"
t00 = time.time(); RECS = [run_seed(s) for s in SEEDS]

# Règles (après coup, à budget égal)
def gate_A(rec, tc): return (rec["T"] > tc) & (rec["G"] > 0)

def topN(S, n):                                           # accepte les n meilleurs slots (score fini) sur toute la seed
    flat = S.ravel(); fin = np.isfinite(flat); n = int(min(n, fin.sum())); A = np.zeros(flat.shape, bool)
    if n > 0: A[np.argsort(-np.where(fin, flat, -np.inf), kind="stable")[:n]] = True
    return A.reshape(S.shape)

def build_rules(rec, rng):
    fin = np.isfinite(rec["T"]); A1 = gate_A(rec, TC_P); N = int(A1.sum())
    R = {"compat_k2": fin.copy(), PROPOSED: A1, STRICT: gate_A(rec, TC_S)}
    SC = {"t": np.where(fin & (rec["G"] > 0), rec["T"], np.nan), "g": np.where(fin, rec["G"], np.nan),
          "cp": np.where(fin, rec["CP"], np.nan), "hg": np.where(fin, rec["HG"], np.nan)}
    for s in SNAMES:
        for mu in MULTS: R[f"top_{s}@{mu}x"] = topN(SC[s], round(mu * N))
    R["rand@1.0x"] = topN(np.where(fin, rng.random(fin.shape), np.nan), N)
    cnt = A1.sum(-1); Gf = np.where(fin, rec["G"], -np.inf)                     # même nombre de sources par client et par couche que le gate
    R["ncount_cp"] = np.stack([cnt > 0, cnt > 1], -1)
    p1 = Gf[..., 1] > Gf[..., 0]; R["ncount_g"] = np.stack([(cnt == 2) | ((cnt == 1) & ~p1), (cnt == 2) | ((cnt == 1) & p1)], -1)
    rd = rng.integers(0, 2, size=cnt.shape) == 1; R["ncount_rand"] = np.stack([(cnt == 2) | ((cnt == 1) & ~rd), (cnt == 2) | ((cnt == 1) & rd)], -1)
    return R

def evaluate(rec, A):
    m1 = A[:, :, 0, 0].astype(int) + 2 * A[:, :, 0, 1]; m2 = A[:, :, 1, 0].astype(int) + 2 * A[:, :, 1, 1]
    dbv = rec["DB"][np.arange(nM)[:, None], np.arange(nK)[None, :], m1, m2]
    u = (A[:, :, 0, :] | A[:, :, 1, :]).sum(-1)
    return dict(SB=np.nansum(dbv, 0), ACC=A.sum((0, 2, 3)).astype(float), POS=(A & (rec["GE"] > 0)).sum((0, 2, 3)).astype(float),
                HARM=(A & (rec["GE"] < -HARM_DELTA)).sum((0, 2, 3)).astype(float), NSRC=A.sum(-1).mean(-1).sum(0),
                COV1=(u >= 1).sum(0).astype(float), COV2=(u >= 2).sum(0).astype(float),
                COMM=sum(A[:, :, li, :].sum((0, 2)) * LSIZE[l] for li, l in enumerate(HIDL)).astype(float))

EV = {r: [] for r in RULE_NAMES}; same_chk = []
for s, rec in zip(SEEDS, RECS):
    R = build_rules(rec, np.random.default_rng(s)); same_chk.append(np.array_equal(R["top_t@1.0x"], R[PROPOSED]))
    for r in RULE_NAMES: EV[r].append(evaluate(rec, R[r]))

# Analyse
SM = nM * len(SEEDS); agg = lambda r, key: sum(e[key] for e in EV[r])
FTB = sum(np.nansum(rec["FTB"], 0) for rec in RECS) / SM
dB = {r: agg(r, "SB") / SM for r in RULE_NAMES}
ACC_ = {r: agg(r, "ACC") for r in RULE_NAMES}; POS_ = {r: agg(r, "POS") for r in RULE_NAMES}; HARM_ = {r: agg(r, "HARM") for r in RULE_NAMES}
NSRC = {r: agg(r, "NSRC") / SM for r in RULE_NAMES}; COV1 = {r: agg(r, "COV1") / SM for r in RULE_NAMES}
COMM = {r: agg(r, "COMM") / SM for r in RULE_NAMES}
HF = {r: np.where(ACC_[r] > 0, HARM_[r] / np.maximum(ACC_[r], 1), 0.0) for r in RULE_NAMES}
HE = {r: HARM_[r] / (SM * NL * K) for r in RULE_NAMES}
sdB = lambda r: [float(e["SB"].mean() / nM) for e in EV[r]]
sHE = lambda r: [float(e["HARM"].mean() / (nM * NL * K)) for e in EV[r]]
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)

def cvar(v, q=CVAR_Q):
    v = np.sort(np.asarray(v, float)); m = max(1, int(np.ceil(q * len(v)))); return v[:m].mean()
def boot(fn, pool=None, B=BOOT_B, seed=1):
    pool = np.arange(nK) if pool is None else np.asarray(pool); rg = np.random.default_rng(seed)
    bs = np.array([fn(pool[rg.integers(0, len(pool), len(pool))]) for _ in range(B)]); return fn(pool), *np.percentile(bs, [2.5, 97.5])
def fm(t, s=1e3, nd=3): return f"{t[0]*s:+.{nd}f} [{t[1]*s:+.{nd}f}, {t[2]*s:+.{nd}f}]"
def pp(r, M_): a = ACC_[r].sum(); return round(M_[r].sum() / a, 3) if a > 0 else np.nan
def contrast(a, b):
    d = dB[a] - dB[b]
    return dict(mean=boot(lambda i: d[i].mean()), dcvar=boot(lambda i: cvar(dB[a][i]) - cvar(dB[b][i])),
                hf=boot(lambda i: (HF[a] - HF[b])[i].mean()), hexp=boot(lambda i: (HE[a] - HE[b])[i].mean()),
                s_mean=sum(x - y > 0 for x, y in zip(sdB(a), sdB(b))), s_hexp=sum(x - y < 0 for x, y in zip(sHE(a), sHE(b))))

print("=" * 150 + f"\nA0. CONTRÔLES D'INTÉGRITÉ [{LEVEL}] seeds={SEEDS}\n" + "=" * 150)
print(f"  top_t@1.0x identique à {PROPOSED} sur toutes les seeds : {all(same_chk)} (attendu : True ; sinon seulement des égalités de score)")
print(f"  Reproduction de EXP-Final (seeds 223-227 : n_src 0.353 | precision_pair 0.898 | harm_pair 0.056) -> ici : n_src {NSRC[PROPOSED].mean():.3f} | "
      f"precision_pair {pp(PROPOSED, POS_)} | harm_pair {pp(PROPOSED, HARM_)} | FT Brier {FTB.mean():.4f}")

MAIN = ["compat_k2", PROPOSED, STRICT, "top_t@1.0x", "top_g@1.0x", "top_cp@1.0x", "top_hg@1.0x", "rand@1.0x", "ncount_cp", "ncount_g", "ncount_rand"]
print("\n" + "=" * 150 + "\nA1. RÈGLES À BUDGET ÉGAL (1.0x = même nombre de paires acceptées que sel@1.645) — dBrier x1e3 (>0 = mieux que FT_k)\n" + "=" * 150)
rows = []
for r in MAIN:
    v = dB[r]
    rows.append(dict(rule=r, n_src=round(NSRC[r].mean(), 3), cov_ge1=round(COV1[r].mean(), 3), precision_pair=pp(r, POS_), harm_pair=pp(r, HARM_),
                     harm_frac_client=fm(boot(lambda j: HF[r][j].mean()), 1, 3), harm_expo=fm(boot(lambda j: HE[r][j].mean()), 1, 3),
                     comm_Mparams=round(COMM[r].mean() / 1e6, 4), dB_mean=fm(boot(lambda i: v[i].mean())), CVaR10=fm(boot(lambda i: cvar(v[i]))),
                     worst=f"{v.min()*1e3:+.3f}", per_seed="/".join(f"{x*1e3:+.2f}" for x in sdB(r))))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 150 + f"\nB. CONTRASTES APPARIÉS À BUDGET ÉGAL : {PROPOSED} - règle (IC95 bootstrap sur les clients)\n"
      "   mean/dCVaR10 en dBrier x1e3 (>0 : gate meilleur) | harm_frac, harm_expo : gate - règle (<0 : gate moins nuisible)\n" + "=" * 150)
rows = []
for b in ["compat_k2", "top_g@1.0x", "top_cp@1.0x", "top_hg@1.0x", "rand@1.0x", "ncount_cp", "ncount_g", "ncount_rand"]:
    c = contrast(PROPOSED, b)
    rows.append(dict(vs=b, mean=fm(c["mean"]), dCVaR10=fm(c["dcvar"]), harm_frac=fm(c["hf"], 1, 3), harm_expo=fm(c["hexp"], 1, 3),
                     seeds_mean_pos=f"{c['s_mean']}/{len(SEEDS)}", seeds_hexp_neg=f"{c['s_hexp']}/{len(SEEDS)}"))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 150 + "\nC. FRONTIÈRE : chaque score classe les paires, on accepte les N x mult meilleures (mult = 0.5, 1, 1.5, 2)\n" + "=" * 150)
rows = []
for s in SNAMES:
    for mu in MULTS:
        r = f"top_{s}@{mu}x"; v = dB[r]
        rows.append(dict(score=s, mult=mu, n_src=round(NSRC[r].mean(), 3), cov_ge1=round(COV1[r].mean(), 3), precision_pair=pp(r, POS_), harm_pair=pp(r, HARM_),
                         harm_expo=f"{HE[r].mean()*1e3:+.2f}", dB_mean=fm(boot(lambda i: v[i].mean())), CVaR10=f"{cvar(v)*1e3:+.3f}"))
print(pd.DataFrame(rows).to_string(index=False))
print("  Lecture : 't' = gate (gain normalisé) ; 'g' = gain brut ; 'cp' = seuil statique sur la compatibilité ; 'hg' = gain moyen historique.")

print("\n" + "=" * 150 + "\nD. t CONTRE CHAQUE ALTERNATIVE, À CHAQUE BUDGET : top_t - top_X (harm_expo < 0 : t plus sûr ; mean > 0 : t meilleur)\n" + "=" * 150)
rows = []
for mu in MULTS:
    for x in ["g", "cp", "hg"]:
        c = contrast(f"top_t@{mu}x", f"top_{x}@{mu}x")
        rows.append(dict(mult=mu, vs=x, mean=fm(c["mean"]), dCVaR10=fm(c["dcvar"]), harm_expo=fm(c["hexp"], 1, 3), seeds_hexp_neg=f"{c['s_hexp']}/{len(SEEDS)}"))
print(pd.DataFrame(rows).to_string(index=False))

# E. verdict (critères fixés avant le run, budget 1.0x)
print("\n" + "=" * 150 + "\nE. VERDICT — le gain normalisé (t) apporte-t-il quelque chose que g, cp ou hg n'apportent pas, à budget égal ?\n" + "=" * 150)
ok_sec, ok_mean = [], []
for x in ["g", "cp", "hg"]:
    c = contrast("top_t@1.0x", f"top_{x}@1.0x")
    p1 = c["hexp"][2] < 0 and c["s_hexp"] >= 4; p2 = c["mean"][1] > -NI_MARGIN; ok_sec.append(p1); ok_mean.append(p2)
    print(f"  [{'PASS' if p1 else 'FAIL'}] (PRIMAIRE) V1[{x}]  harm_expo(t) - harm_expo({x}) < 0 (IC95 < 0) et < 0 sur >= 4/5 seeds  ->  {fm(c['hexp'],1,3)} | seeds {c['s_hexp']}/{len(SEEDS)}")
    print(f"  [{'PASS' if p2 else 'FAIL'}]             V2[{x}]  mean(t - {x}) non-inférieur : borne basse IC95 > -{NI_MARGIN*1e3:.2f}e-3  ->  {fm(c['mean'])}")
    print(f"  [INFO]              dCVaR10(t - {x}) = {fm(c['dcvar'])}")
cr = contrast("top_t@1.0x", "rand@1.0x"); cg = contrast("top_t@1.0x", "ncount_g")
print(f"  [INFO] t vs rand@1.0x : mean {fm(cr['mean'])} | harm_expo {fm(cr['hexp'],1,3)}")
print(f"  [INFO] t vs ncount_g (même nombre par client/couche, classement par gain brut) : mean {fm(cg['mean'])} | harm_expo {fm(cg['hexp'],1,3)}")
if all(ok_sec) and all(ok_mean):
    print("  => t DOMINE à budget égal : la normalisation par l'erreur-type apporte une sélection plus sûre que gain brut, compatibilité et gain historique. "
          "Claim : 'normalized-evidence gate'. Ne dites PAS 'e-process' (pas de validité séquentielle).")
elif any(ok_sec):
    print("  => PARTIEL : t bat certaines alternatives seulement (voir V1 par score). Claim limité à celles-ci ; les autres restent des concurrents équivalents.")
else:
    print("  => NON ÉTABLI : à budget égal, t n'est pas plus sûr que les règles simples. Le gate reste un mécanisme d'abstention, mais sa forme n'est pas justifiée "
          "par ces données. Présentez 'abstention par seuil sur un score de transfert' sans revendiquer la supériorité du test.")
print(f"\nTotal time: {(time.time() - t00) / 60:.1f} min")
