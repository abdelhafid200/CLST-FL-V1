# CLST-FL : EXP-Baselines-2, baselines réglées (grille de lambda pour Ditto, arrêt anticipé sur la validation pour les modèles persistants)
# Mêmes clients, même test, même référence FT_k que EXP-Baselines. Aucun fichier écrit.
# Ce réglage avantage les baselines : pour local_only / fedper / ditto, le snapshot est choisi par client et par seed sur le Brier de
# validation (parmi 9), alors que le gate n'a pas ce choix. Le test n'est jamais utilisé pour choisir.
# Critères de lecture fixés avant le run (section D).
import os, json, pickle, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config
LEVEL = "high"
BASE_DIR = "data_v3"   # dossier écrit par data_00_build_clients_v3.py ; à adapter
CLIENT_DIR = os.path.join(BASE_DIR, f"clients_hetero_{LEVEL}")
SEEDS = [423, 424, 425]                  # mêmes seeds que EXP-Baselines (contrôle de reproduction) ; ajouter 426, 427 si le temps le permet
N_ROUNDS, WARMUP, MEAS_EVERY = 50, 3, 5
MEAS_ROUNDS = list(range(WARMUP + MEAS_EVERY, N_ROUNDS + 1, MEAS_EVERY)); nM = len(MEAS_ROUNDS)
LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE = 5, 1e-3, 64
HIDDEN, DROPOUT = (64, 32), 0.10
K, GAMMA = 2, 0.10
TC_P, TC_S = 1.645, 2.0
DITTO_LAMS = [0.03, 0.1, 0.3]            # lambda choisi par le Brier de validation (au snapshot retenu), jamais par le test
HARM_DELTA = 1e-4
HC_ANY, HC_MAT = 1e-5, 1e-4              # "pire que FT" : tout écart (>0.01e-3) / matériel (>0.1e-3)
UNCH_TOL = 1e-7                          # |dBrier| < tol : modèle inchangé par rapport à FT
BOOT_B, CVAR_Q, EPS = 2000, 0.10, 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDL = ["layer1", "layer2"]
PROPOSED, STRICT = f"sel@{TC_P}", f"sel@{TC_S}"
OURS = ["compat_k2", PROPOSED, STRICT]
FREE = ["fedavg_global", "fomo_orig", "fomo_upd"]                       # calculés sans entraînement supplémentaire
PERSIST = ["local_only", "fedper"] + [f"ditto_l{l}" for l in DITTO_LAMS]  # état personnel persistant -> arrêt anticipé sur la validation
RULES = OURS + FREE + PERSIST; RIDX = {n: i for i, n in enumerate(RULES)}; nR = len(RULES)
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

def train_local(start_vec, cid, seed, epochs=LOCAL_EPOCHS, center=None, lam=0.0):
    """Adam sur le Brier ; si center est donné : + (lam/2)*||theta - center||^2 (terme proximal de Ditto)."""
    torch.manual_seed(int(seed))
    model = SimpleMLP().to(DEVICE)
    vector_to_parameters(start_vec.clone().to(DEVICE), model.parameters()); model.train()
    c = None if center is None else center.clone().to(DEVICE)
    X, y = data[cid]["Xtr_dev"], data[cid]["ytr_dev"]
    opt = torch.optim.Adam(model.parameters(), lr=LOCAL_LR); n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]; opt.zero_grad()
            loss = ((torch.sigmoid(model(X[idx])) - y[idx]) ** 2).mean()
            if c is not None and lam > 0: loss = loss + 0.5 * lam * ((parameters_to_vector(model.parameters()) - c) ** 2).sum()
            loss.backward(); opt.step()
    return parameters_to_vector(model.parameters()).detach().cpu().clone()

def fedavg(vecs): return sum((n_train[k] / N_TOTAL) * vecs[k] for k in IDS)

def pmask(idxs):
    m, o = torch.zeros(P), 0
    for i, sz in enumerate(_SIZES):
        if i in idxs: m[o:o + sz] = 1.0
        o += sz
    return m
CMASK = dict(layer1=pmask({0, 1}), layer2=pmask({2, 3}), head=pmask({4, 5}))
HIDMASK = CMASK["layer1"] + CMASK["layer2"]

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
      f"Ditto lambdas={DITTO_LAMS} | {nM} snapshots/seed | règles: {RULES}")

def make_splits():
    return ({cid: (data[cid]["Xva"], data[cid]["yva"]) for cid in IDS}, {cid: (data[cid]["Xte"], data[cid]["yte"]) for cid in IDS})

def topk_mask(score, allowed, k=K):
    out = np.zeros(len(allowed), bool); idx = np.where(allowed)[0]
    if len(idx) and k > 0: out[idx[np.argsort(-score[idx])[:k]]] = True
    return out

# Mesure
def new_rec():
    return dict(DB={r: [] for r in RULES}, VB={r: [] for r in PERSIST}, FTB=[],
                ACC={r: np.zeros(nK) for r in OURS}, POS={r: np.zeros(nK) for r in OURS}, HARM={r: np.zeros(nK) for r in OURS},
                NSRC={r: np.zeros(nK) for r in OURS}, COV1={r: np.zeros(nK) for r in OURS}, M=0)

def measure(FTt, Gprev, Gnew, DVs, LVs, FPT, sel, ev, rec):
    rec["M"] += 1; Dt = FTt - Gprev[None]
    V = {l: Dt * CMASK[l] for l in HIDL}; Vh = Dt * HIDMASK
    db = {r: np.zeros(nK) for r in RULES}; vb = {r: np.zeros(nK) for r in PERSIST}; ftb = np.zeros(nK)
    pidx = [RIDX[n] for n in PERSIST]
    for ki, k in enumerate(IDS):
        Xs, ys = sel[k]; Xe, ye = ev[k]; ns = len(ys)
        ys = np.asarray(ys, float); ye = np.asarray(ye, float)
        Xc, yc = torch.cat([Xs, Xe]), np.concatenate([ys, ye])
        Lb = (batch_probs(FTt[ki][None], Xc)[0] - yc) ** 2; ftb[ki] = Lb[ns:].mean()
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
        plans = {"compat_k2": {l: cand for l in HIDL}, PROPOSED: {l: cand & gate[(l, TC_P)] for l in HIDL}, STRICT: {l: cand & gate[(l, TC_S)] for l in HIDL}}
        vecs = []
        for n in OURS:
            v = FTt[ki].clone()
            for l, m in plans[n].items():
                if m.any(): v = v + GAMMA * (torch.tensor(m / m.sum(), dtype=torch.float32) @ V[l])
            vecs.append(v)
            masks = list(plans[n].values())
            for l, m in plans[n].items():
                g_ = gE[l][m]; rec["ACC"][n][ki] += m.sum(); rec["POS"][n][ki] += (g_ > 0).sum(); rec["HARM"][n][ki] += (g_ < -HARM_DELTA).sum()
            rec["NSRC"][n][ki] += np.mean([m.sum() for m in masks]); rec["COV1"][n][ki] += (np.logical_or.reduce(masks).sum() >= 1)
        # FedFomo : poids = max(0, gain de perte de validation) / distance des paramètres, normalisés
        dist = torch.norm(FTt - FTt[ki][None], dim=1).numpy()
        w = np.where(valid, np.maximum(cp, 0.0) / (dist + EPS), 0.0); s_ = w.sum()
        if s_ > 0:
            wt = torch.tensor(w / s_, dtype=torch.float32)
            fo, fu = FTt[ki] + wt @ (FTt - FTt[ki][None]), FTt[ki] + GAMMA * (wt @ Vh)
        else: fo, fu = FTt[ki].clone(), FTt[ki].clone()
        vecs += [Gnew, fo, fu, LVs[k], FPT[k]] + [DVs[l][k] for l in DITTO_LAMS]
        assert len(vecs) == nR
        Pr = batch_probs(torch.stack(vecs), Xe); bt = ((Pr - ye[None]) ** 2).mean(1)
        for i, n in enumerate(RULES): db[n][ki] = ftb[ki] - bt[i]
        Pv = batch_probs(torch.stack([vecs[i] for i in pidx]), Xs)                  # Brier de validation des modèles persistants
        for j, n in enumerate(PERSIST): vb[n][ki] = ((Pv[j] - ys) ** 2).mean()
    for r in RULES: rec["DB"][r].append(db[r])
    for r in PERSIST: rec["VB"][r].append(vb[r])
    rec["FTB"].append(ftb)

def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    G0 = parameters_to_vector(SimpleMLP().parameters()).detach().cpu().clone(); G = G0.clone()
    DV = {l: {k: G0.clone() for k in IDS} for l in DITTO_LAMS}      # modèles personnels Ditto (état persistant)
    LV = {k: G0.clone() for k in IDS}                               # local_only (état persistant, sans fédération)
    HEAD = {k: G0.clone() for k in IDS}; FPG = G0.clone()           # FedPer : corps partagé, tête locale persistante
    hm = CMASK["head"]; sd = lambda r, k, tag: seed + r * 1000 + k + tag * 10_000_000
    rec, t0 = new_rec(), time.time()
    for r in range(1, N_ROUNDS + 1):
        FT = {k: train_local(G, k, sd(r, k, 0)) for k in IDS}; Gnew = fedavg(FT)
        for i, l in enumerate(DITTO_LAMS):
            for k in IDS: DV[l][k] = train_local(DV[l][k], k, sd(r, k, 1 + i), center=G, lam=l)
        for k in IDS: LV[k] = train_local(LV[k], k, sd(r, k, 5))
        FPT = {k: train_local(FPG * (1 - hm) + HEAD[k] * hm, k, sd(r, k, 6)) for k in IDS}
        for k in IDS: HEAD[k] = FPT[k]
        FPG = fedavg(FPT) * (1 - hm) + FPG * hm
        if r in MEAS_ROUNDS: measure(torch.stack([FT[k] for k in IDS]), G, Gnew, DV, LV, FPT, *make_splits(), rec)
        G = Gnew
    print(f"[seed={seed}] terminé en {(time.time() - t0) / 60:.1f} min")
    return rec

_V = torch.stack([parameters_to_vector(SimpleMLP().parameters()).detach() for _ in range(3)]); _X = data[IDS[0]]["Xva"][:64]
assert np.allclose(batch_probs(_V, _X)[1], probs_of(_V[1], _X), atol=1e-5), "batch_probs != probs_of"
t00 = time.time(); RECS = [run_seed(s) for s in SEEDS]

# Analyse
S = len(SEEDS)
DB_ALL = {r: np.array([np.array(rec["DB"][r]) for rec in RECS]) for r in RULES}          # (S, nM, nK)
FTBA = np.array([np.array(rec["FTB"]) for rec in RECS])                                  # (S, nM, nK)
FTB = FTBA.reshape(-1, nK).mean(0)
D_EFF, FT_REF, VALSEL, MSTAR = {}, {}, {}, {}
for r in RULES:
    if r in PERSIST:                                                                     # snapshot choisi par client et par seed sur la validation
        VB = np.array([np.array(rec["VB"][r]) for rec in RECS]); ms = VB.argmin(1)       # (S, nK)
        D_EFF[r] = np.take_along_axis(DB_ALL[r], ms[:, None, :], 1)[:, 0, :]             # (S, nK)
        FT_REF[r] = float(np.take_along_axis(FTBA, ms[:, None, :], 1)[:, 0, :].mean()); VALSEL[r] = float(VB.min(1).mean()); MSTAR[r] = ms
    else:
        D_EFF[r] = DB_ALL[r].reshape(S * nM, nK); FT_REF[r] = float(FTB.mean())
dB = {r: D_EFF[r].mean(0) for r in RULES}
dB_all = {r: DB_ALL[r].reshape(S * nM, nK).mean(0) for r in RULES}                       # moyenne sur tous les snapshots (sans arrêt anticipé)
HW_ANY = {r: (D_EFF[r] < -HC_ANY).mean(0) for r in RULES}; HW_MAT = {r: (D_EFF[r] < -HC_MAT).mean(0) for r in RULES}
UNCH = {r: float((np.abs(DB_ALL[r]) < UNCH_TOL).mean()) for r in RULES}
sdB = lambda r: D_EFF[r].mean(1) if r in PERSIST else DB_ALL[r].mean((1, 2))
sdB_all = lambda r: DB_ALL[r].mean((1, 2))
LAM_STAR = min(DITTO_LAMS, key=lambda l: VALSEL[f"ditto_l{l}"]); DITTO_SEL = f"ditto_l{LAM_STAR}"
VERDICT_BASES = ["fedavg_global", "local_only", "fedper", "fomo_upd", DITTO_SEL]
pd.set_option("display.width", 340); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)

def cvar(v, q=CVAR_Q):
    v = np.sort(np.asarray(v, float)); m = max(1, int(np.ceil(q * len(v)))); return v[:m].mean()
def boot(fn, B=BOOT_B, seed=1):
    rg = np.random.default_rng(seed); bs = np.array([fn(rg.integers(0, nK, nK)) for _ in range(B)])
    return fn(np.arange(nK)), *np.percentile(bs, [2.5, 97.5])
def fm(t, s=1e3, nd=3): return f"{t[0]*s:+.{nd}f} [{t[1]*s:+.{nd}f}, {t[2]*s:+.{nd}f}]"
def mark(r): return r + ("*" if r == DITTO_SEL else "")

def contrast(a, b):
    d = dB[a] - dB[b]
    return dict(mean=boot(lambda i: d[i].mean()), dcvar=boot(lambda i: cvar(dB[a][i]) - cvar(dB[b][i])),
                hmat=boot(lambda i: (HW_MAT[a] - HW_MAT[b])[i].mean()), hany=boot(lambda i: (HW_ANY[a] - HW_ANY[b])[i].mean()),
                s_mean=int(sum(x - y > 0 for x, y in zip(sdB(a), sdB(b)))))

def label(c):
    m = "+" if c["mean"][1] > 0 else "-" if c["mean"][2] < 0 else "="
    s = "safer" if c["hmat"][2] < 0 else "riskier" if c["hmat"][1] > 0 else "="
    t = "better" if c["dcvar"][1] > 0 else "worse" if c["dcvar"][2] < 0 else "="
    if m == "+" and s != "riskier": return "GATE DOMINE", (m, s, t)
    if m == "+": return "COMPROMIS : gate plus précis, baseline plus sûre", (m, s, t)
    if m == "-" and s != "safer" and t != "better": return "BASELINE DOMINE", (m, s, t)
    if m == "-": return "COMPROMIS : baseline plus précise, gate plus sûr / meilleure queue", (m, s, t)
    if s == "safer": return "MÊME PRÉCISION, GATE PLUS SÛR", (m, s, t)
    if s == "riskier": return "MÊME PRÉCISION, BASELINE PLUS SÛRE", (m, s, t)
    return "INDISTINCT", (m, s, t)

print("=" * 160 + f"\nA0. CONTRÔLES [{LEVEL}] seeds={SEEDS} | FT Brier test moyen={FTB.mean():.4f}\n" + "=" * 160)
print("  Brier de VALIDATION moyen des modèles persistants au snapshot retenu : " + " | ".join(f"{r}: {VALSEL[r]:.5f}" for r in PERSIST))
print(f"  Ditto : lambda retenu sur la validation = {LAM_STAR} ({DITTO_SEL}*)" +
      ("  | ATTENTION : au bord de la grille, élargir avant toute conclusion sur Ditto." if LAM_STAR in (min(DITTO_LAMS), max(DITTO_LAMS)) else "  | optimum intérieur à la grille."))
for r in PERSIST:
    ms = MSTAR[r]; print(f"  snapshot retenu pour {r:<12} : round moyen {np.mean(np.array(MEAS_ROUNDS)[ms]):5.1f} | part des clients au dernier snapshot {np.mean(ms == nM - 1):.2f} | au premier {np.mean(ms == 0):.2f}")
agg = lambda key, r: sum(rec[key][r] for rec in RECS)
a_ = agg("ACC", PROPOSED).sum()
print(f"  Reproduction {PROPOSED} (EXP-Baselines, seeds 423-425 : n_src 0.361 | precision_pair 0.896 | harm_pair 0.049) -> ici : n_src {(agg('NSRC', PROPOSED) / (S * nM)).mean():.3f} | "
      f"precision_pair {agg('POS', PROPOSED).sum() / a_:.3f} | harm_pair {agg('HARM', PROPOSED).sum() / a_:.3f}")
print("  Reproduction par seed, moyenne sur TOUS les snapshots, x1e3 (EXP-Baselines 423/424/425 : sel@1.645 +0.17/+0.17/+0.13 | fomo_upd +0.16/+0.14/+0.10 | fedper +0.29/+0.17/+0.09) -> ici : "
      + " | ".join(f"{r} " + "/".join(f"{x*1e3:+.2f}" for x in sdB_all(r)) for r in (PROPOSED, "fomo_upd", "fedper")))

print("\n" + "=" * 160 + "\nA. PERFORMANCE ET SÉCURITÉ — dBrier x1e3 vs FT_k (>0 = mieux que FT). Modèles persistants : snapshot choisi sur la VALIDATION ; autres règles : moyenne sur les snapshots\n" + "=" * 160)
rows = [dict(rule="FT (= FedAvg + FT local)", dB_mean="+0.000", rel="0%", dB_allsnap="-", CVaR10="0", worst="0", unchanged="-", harm_any="-", harm_mat="-", per_seed="0")]
for r in RULES:
    v = dB[r]
    rows.append(dict(rule=mark(r), dB_mean=fm(boot(lambda i: v[i].mean())), rel=f"{v.mean() / FT_REF[r]:+.2%}",
                     dB_allsnap=f"{dB_all[r].mean()*1e3:+.3f}" if r in PERSIST else "(idem)",
                     CVaR10=fm(boot(lambda i: cvar(v[i]))), worst=f"{v.min()*1e3:+.3f}", unchanged=f"{UNCH[r]:.2f}",
                     harm_any=fm(boot(lambda i: HW_ANY[r][i].mean()), 1, 3), harm_mat=fm(boot(lambda i: HW_MAT[r][i].mean()), 1, 3),
                     per_seed="/".join(f"{x*1e3:+.2f}" for x in sdB(r))))
print(pd.DataFrame(rows).to_string(index=False))
print("  * = lambda de Ditto retenu. dB_allsnap = même règle sans arrêt anticipé (moyenne des 9 snapshots). unchanged = part des (seed, snapshot, client) où le modèle est identique à FT :")
print("  une sécurité élevée avec unchanged élevé est en partie mécanique (un modèle inchangé ne peut pas être pire que FT).")

print("\n" + "=" * 160 + f"\nB. CONTRASTES APPARIÉS : {PROPOSED} - baseline (IC95 bootstrap sur les 48 clients) — mean/dCVaR10 x1e3 (>0 : gate meilleur) ; harm : gate - baseline (<0 : gate plus sûr)\n" + "=" * 160)
rows = []
for b in ["compat_k2"] + FREE + PERSIST:
    c = contrast(PROPOSED, b); lab, _ = label(c)
    rows.append(dict(vs=mark(b), mean=fm(c["mean"]), dCVaR10=fm(c["dcvar"]), harm_mat=fm(c["hmat"], 1, 3), harm_any=fm(c["hany"], 1, 3),
                     seeds_mean_pos=f"{c['s_mean']}/{S}", lecture=lab))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 160 + "\nC. PAR TAILLE DE CLIENT (Q1 = 12 plus petits ... Q4 = 12 plus grands) — dBrier x1e3 moyen par quartile\n" + "=" * 160)
SIZEQ = pd.qcut(pd.Series(NTR).rank(method="first"), 4, labels=False).values
show = ["compat_k2", PROPOSED, "fedavg_global", "local_only", "fedper", "fomo_upd", DITTO_SEL]
print(pd.DataFrame({mark(r): [f"{dB[r][SIZEQ == q].mean()*1e3:+.3f}" for q in range(4)] for r in show}, index=[f"Q{q+1}" for q in range(4)]).to_string())

print("\n" + "=" * 160 + "\nD. LECTURE (règles fixées avant le run ; le verdict porte sur les 5 baselines ci-dessous, Ditto au lambda retenu sur la validation)\n" + "=" * 160)
print("  m (précision) : '+' si l'IC95 de mean(gate - X) est > 0, '-' s'il est < 0, '=' sinon.  s (sécurité) : 'safer' si l'IC95 de harm_mat(gate - X) est < 0, 'riskier' s'il est > 0.")
print("  t (queue) : 'better' si l'IC95 de dCVaR10 est > 0, 'worse' s'il est < 0.")
res = {}
for b in VERDICT_BASES:
    c = contrast(PROPOSED, b); lab, (m, s, t) = label(c); res[b] = (lab, m, s, t, c)
    print(f"  {mark(b):<16} m={m} s={s:<7} t={t:<6} -> {lab}")
better = [b for b in VERDICT_BASES if res[b][1] == "-"]
same_safer = [b for b in VERDICT_BASES if res[b][1] == "=" and res[b][2] == "safer"]
gate_better = [b for b in VERDICT_BASES if res[b][1] == "+"]
print(f"\n  Plus précises que le gate (IC95 < 0) : {better if better else 'aucune'}")
print(f"  Précision indistincte et gate plus sûr : {same_safer if same_safer else 'aucune'}")
print(f"  Gate plus précis (IC95 > 0) : {gate_better if gate_better else 'aucune'}")
cf = res["fomo_upd"][4]
print(f"  [INFO] concurrent le plus proche, fomo_upd : mean {fm(cf['mean'])} | harm_mat {fm(cf['hmat'],1,3)} | harm_any {fm(cf['hany'],1,3)} | unchanged gate {UNCH[PROPOSED]:.2f} vs fomo_upd {UNCH['fomo_upd']:.2f}")
if better:
    print("  => POSITIONNEMENT À REVOIR : au moins une baseline réglée est significativement plus précise que le gate. Reformuler (compromis précision/sécurité) avant d'écrire l'Introduction.")
else:
    print("  => Aucune baseline réglée sur la validation n'est significativement plus précise que le gate. Le positionnement 'précision comparable, moins de clients dégradés' tient, "
          "avec les réserves : référence FT commune, sécurité en partie mécanique (voir 'unchanged'), traitement favorable aux baselines (snapshot choisi sur la validation).")
print(f"\nTotal time: {(time.time() - t00) / 60:.1f} min")
