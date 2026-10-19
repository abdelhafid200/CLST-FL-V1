# CLST-FL : EXP-Fallback. Travailler au niveau paire x couche apporte-t-il quelque chose par rapport a un choix au niveau du modele entier ?
# Baselines : fb_fedper_{t,raw}, fb_ditto_{t,raw} = a chaque snapshot, le client choisit FT_k ou son modele persistant (FedPer / Ditto lambda=0.03)
#   _raw : si le Brier de validation du modele persistant < celui de FT_k
#   _t   : meme statistique que notre gate (t > 1.645 et gain > 0) appliquee au modele entier
# Nouvelles seeds (523-525). Decision prise a chaque snapshot (comme pour le gate), sans arret anticipe ; le test ne sert jamais a choisir.
# N'ecrit aucun fichier. Criteres fixes avant le run (section D).
import os, json, pickle, random, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config
LEVEL = "high"
BASE_DIR = "data_v3"   # dossier écrit par data_00_build_clients_v3.py ; a adapter
CLIENT_DIR = os.path.join(BASE_DIR, f"clients_hetero_{LEVEL}")
SEEDS = [523, 524, 525]                  # pas encore utilisees ; ajouter 526, 527 si le temps le permet
N_ROUNDS, WARMUP, MEAS_EVERY = 50, 3, 5
MEAS_ROUNDS = list(range(WARMUP + MEAS_EVERY, N_ROUNDS + 1, MEAS_EVERY)); nM = len(MEAS_ROUNDS)
LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE = 5, 1e-3, 64
HIDDEN, DROPOUT = (64, 32), 0.10
K, GAMMA = 2, 0.10
TC_P, TC_S = 1.645, 2.0
DITTO_LAM = 0.03                         # retenu sur la validation dans EXP-Baselines-2
HARM_DELTA = 1e-4                        # nuisible au niveau paire
HC_MAT, HC_BIG = 1e-4, 1e-3              # client "materiellement pire que FT" : > 0.1e-3 / > 1e-3
UNCH_TOL = 1e-7
BOOT_B, CVAR_Q, EPS = 2000, 0.10, 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDL = ["layer1", "layer2"]
PROPOSED, STRICT = f"sel@{TC_P}", f"sel@{TC_S}"
OURS = ["compat_k2", PROPOSED, STRICT]
DIRECT = ["fomo_upd", "fedper", "ditto_l0.03"]                 # evalues directement sur le test
FB = {"fb_fedper_t": ("fedper", "t"), "fb_fedper_raw": ("fedper", "raw"),
      "fb_ditto_t": ("ditto_l0.03", "t"), "fb_ditto_raw": ("ditto_l0.03", "raw")}
FALL = list(FB)
RULES = OURS + DIRECT + FALL; RIDX = {n: i for i, n in enumerate(RULES)}
CAND_IDX = {"fedper": 0, "ditto_l0.03": 1}
with open(os.path.join(CLIENT_DIR, "metadata.json"), "r") as f: META = json.load(f)
INPUT_DIM = META["input_dim"]

# Modele
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
    """Adam sur le Brier ; si center est donne : + (lam/2)*||theta - center||^2 (terme proximal de Ditto)."""
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

# Donnees
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
      f"{nM} snapshots/seed | regles: {RULES}")

def make_splits():
    return ({cid: (data[cid]["Xva"], data[cid]["yva"]) for cid in IDS}, {cid: (data[cid]["Xte"], data[cid]["yte"]) for cid in IDS})

def topk_mask(score, allowed, k=K):
    out = np.zeros(len(allowed), bool); idx = np.where(allowed)[0]
    if len(idx) and k > 0: out[idx[np.argsort(-score[idx])[:k]]] = True
    return out

# Mesure
def new_rec():
    return dict(DB={r: [] for r in RULES}, FTB=[], ACCF={r: [] for r in FALL},
                ACC={r: np.zeros(nK) for r in OURS}, POS={r: np.zeros(nK) for r in OURS}, HARM={r: np.zeros(nK) for r in OURS},
                NSRC={r: np.zeros(nK) for r in OURS}, M=0)

def measure(FTt, Gprev, DV, FPT, sel, ev, rec):
    rec["M"] += 1; Dt = FTt - Gprev[None]
    V = {l: Dt * CMASK[l] for l in HIDL}; Vh = Dt * HIDMASK
    db = {r: np.zeros(nK) for r in RULES}; accf = {r: np.zeros(nK) for r in FALL}; ftb = np.zeros(nK)
    base = OURS + DIRECT
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
            for l, m in plans[n].items():
                g_ = gE[l][m]; rec["ACC"][n][ki] += m.sum(); rec["POS"][n][ki] += (g_ > 0).sum(); rec["HARM"][n][ki] += (g_ < -HARM_DELTA).sum()
            rec["NSRC"][n][ki] += np.mean([m.sum() for m in plans[n].values()])
        # FedFomo applique avec notre operateur : poids = max(0, gain de validation) / distance des parametres
        dist = torch.norm(FTt - FTt[ki][None], dim=1).numpy()
        w = np.where(valid, np.maximum(cp, 0.0) / (dist + EPS), 0.0); s_ = w.sum()
        fu = FTt[ki] + GAMMA * (torch.tensor(w / s_, dtype=torch.float32) @ Vh) if s_ > 0 else FTt[ki].clone()
        vecs += [fu, FPT[k], DV[k]]
        assert len(vecs) == len(base)
        Pr = batch_probs(torch.stack(vecs), Xe); bt = ((Pr - ye[None]) ** 2).mean(1)
        for i, n in enumerate(base): db[n][ki] = ftb[ki] - bt[i]
        # Baselines "fallback" : choix FT / modele persistant sur la validation, au niveau du modele entier
        pFT = batch_probs(FTt[ki][None], Xs)[0]; Pc = batch_probs(torch.stack([FPT[k], DV[k]]), Xs)
        for n, (cn, mode) in FB.items():
            d = (pFT - ys) ** 2 - (Pc[CAND_IDX[cn]] - ys) ** 2
            g = d.mean(); t = g / (d.std(ddof=1) / np.sqrt(ns) + EPS)
            ok = (g > 0) if mode == "raw" else bool((t > TC_P) and (g > 0))
            accf[n][ki] = float(ok); db[n][ki] = db[cn][ki] if ok else 0.0
    for r in RULES: rec["DB"][r].append(db[r])
    for r in FALL: rec["ACCF"][r].append(accf[r])
    rec["FTB"].append(ftb)

def run_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    G0 = parameters_to_vector(SimpleMLP().parameters()).detach().cpu().clone(); G = G0.clone()
    DV = {k: G0.clone() for k in IDS}                               # Ditto (etat personnel persistant)
    HEAD = {k: G0.clone() for k in IDS}; FPG = G0.clone()           # FedPer : corps partage, tete locale persistante
    hm = CMASK["head"]; sd = lambda r, k, tag: seed + r * 1000 + k + tag * 10_000_000
    rec, t0 = new_rec(), time.time()
    for r in range(1, N_ROUNDS + 1):
        FT = {k: train_local(G, k, sd(r, k, 0)) for k in IDS}; Gnew = fedavg(FT)
        for k in IDS: DV[k] = train_local(DV[k], k, sd(r, k, 1), center=G, lam=DITTO_LAM)
        FPT = {k: train_local(FPG * (1 - hm) + HEAD[k] * hm, k, sd(r, k, 6)) for k in IDS}
        for k in IDS: HEAD[k] = FPT[k]
        FPG = fedavg(FPT) * (1 - hm) + FPG * hm
        if r in MEAS_ROUNDS: measure(torch.stack([FT[k] for k in IDS]), G, DV, FPT, *make_splits(), rec)
        G = Gnew
    print(f"[seed={seed}] termine en {(time.time() - t0) / 60:.1f} min")
    return rec

_V = torch.stack([parameters_to_vector(SimpleMLP().parameters()).detach() for _ in range(3)]); _X = data[IDS[0]]["Xva"][:64]
assert np.allclose(batch_probs(_V, _X)[1], probs_of(_V[1], _X), atol=1e-5), "batch_probs != probs_of"
t00 = time.time(); RECS = [run_seed(s) for s in SEEDS]

# Analyse
S = len(SEEDS); SM = S * nM
D = {r: np.concatenate([np.array(rec["DB"][r]) for rec in RECS], 0) for r in RULES}        # (S*nM, nK)
FTB = np.concatenate([np.array(rec["FTB"]) for rec in RECS], 0).mean(0)
dB = {r: D[r].mean(0) for r in RULES}
HW_MAT = {r: (D[r] < -HC_MAT).mean(0) for r in RULES}; HW_BIG = {r: (D[r] < -HC_BIG).mean(0) for r in RULES}
UNCH = {r: float((np.abs(D[r]) < UNCH_TOL).mean()) for r in RULES}
USED = {r: float(np.mean([np.array(rec["ACCF"][r]).mean() for rec in RECS])) for r in FALL}
sdB = lambda r: D[r].reshape(S, nM, nK).mean((1, 2))
pd.set_option("display.width", 340); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)

def cvar(v, q=CVAR_Q):
    v = np.sort(np.asarray(v, float)); m = max(1, int(np.ceil(q * len(v)))); return v[:m].mean()
def boot(fn, B=BOOT_B, seed=1):
    rg = np.random.default_rng(seed); bs = np.array([fn(rg.integers(0, nK, nK)) for _ in range(B)])
    return fn(np.arange(nK)), *np.percentile(bs, [2.5, 97.5])
def fm(t, s=1e3, nd=3): return f"{t[0]*s:+.{nd}f} [{t[1]*s:+.{nd}f}, {t[2]*s:+.{nd}f}]"

def contrast(a, b):
    d = dB[a] - dB[b]
    return dict(mean=boot(lambda i: d[i].mean()), dcvar=boot(lambda i: cvar(dB[a][i]) - cvar(dB[b][i])),
                hmat=boot(lambda i: (HW_MAT[a] - HW_MAT[b])[i].mean()), hbig=boot(lambda i: (HW_BIG[a] - HW_BIG[b])[i].mean()),
                s_mean=int(sum(x - y > 0 for x, y in zip(sdB(a), sdB(b)))))

def label(c):
    m = "+" if c["mean"][1] > 0 else "-" if c["mean"][2] < 0 else "="
    s = "safer" if c["hmat"][2] < 0 else "riskier" if c["hmat"][1] > 0 else "="
    t = "better" if c["dcvar"][1] > 0 else "worse" if c["dcvar"][2] < 0 else "="
    return m, s, t

print("=" * 160 + f"\nA0. CONTROLES [{LEVEL}] seeds={SEEDS} | FT Brier test moyen={FTB.mean():.4f}\n" + "=" * 160)
agg = lambda key, r: sum(rec[key][r] for rec in RECS); a_ = agg("ACC", PROPOSED).sum()
print(f"  {PROPOSED} sur seeds NEUVES (attendu a peu pres : n_src ~0.35 | precision_pair ~0.88-0.90 | harm_pair ~0.05-0.06) -> ici : n_src {(agg('NSRC', PROPOSED) / SM).mean():.3f} | "
      f"precision_pair {agg('POS', PROPOSED).sum() / a_:.3f} | harm_pair {agg('HARM', PROPOSED).sum() / a_:.3f}")
print(f"  compat_k2 harm_pair (attendu ~0.26) -> ici : {agg('HARM', 'compat_k2').sum() / agg('ACC', 'compat_k2').sum():.3f}")
print("  Part des (snapshot, client) ou le modele persistant est choisi a la place de FT : " + " | ".join(f"{r}: {USED[r]:.2f}" for r in FALL))

print("\n" + "=" * 160 + "\nA. PERFORMANCE ET SECURITE — dBrier x1e3 vs FT_k (>0 = mieux que FT). Toutes les regles : moyenne sur les snapshots, 3 seeds x 9 snapshots par client\n" + "=" * 160)
rows = [dict(rule="FT (= FedAvg + FT local)", dB_mean="+0.000", rel="0%", CVaR10="0", worst="0", worst_rel="0%", unchanged="-", harm_mat="-", harm_big="-", per_seed="0")]
for r in RULES:
    v = dB[r]
    rows.append(dict(rule=r, dB_mean=fm(boot(lambda i: v[i].mean())), rel=f"{v.mean() / FTB.mean():+.2%}", CVaR10=fm(boot(lambda i: cvar(v[i]))),
                     worst=f"{v.min()*1e3:+.3f}", worst_rel=f"{(v / FTB).min():+.2%}", unchanged=f"{UNCH[r]:.2f}",
                     harm_mat=fm(boot(lambda i: HW_MAT[r][i].mean()), 1, 3), harm_big=fm(boot(lambda i: HW_BIG[r][i].mean()), 1, 3),
                     per_seed="/".join(f"{x*1e3:+.2f}" for x in sdB(r))))
print(pd.DataFrame(rows).to_string(index=False))
print("  harm_mat : part des evaluations ou le modele est pire que FT de plus de 0.1e-3 ; harm_big : de plus de 1e-3. worst_rel : pire client, en % du Brier de FT de ce client.")
print("  unchanged : part des evaluations identiques a FT (une securite elevee avec unchanged eleve est en partie mecanique).")

print("\n" + "=" * 160 + f"\nB. CONTRASTES APPARIES : {PROPOSED} - regle (IC95 bootstrap sur les 48 clients) — mean/dCVaR10 x1e3 (>0 : gate meilleur) ; harm : gate - regle (<0 : gate plus sur)\n" + "=" * 160)
rows = []
for b in ["compat_k2", "fomo_upd", "fedper", "ditto_l0.03"] + FALL:
    c = contrast(PROPOSED, b); m, s, t = label(c)
    rows.append(dict(vs=b, mean=fm(c["mean"]), dCVaR10=fm(c["dcvar"]), harm_mat=fm(c["hmat"], 1, 3), harm_big=fm(c["hbig"], 1, 3),
                     seeds_mean_pos=f"{c['s_mean']}/{S}", m=m, s=s, t=t))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 160 + "\nC. PAR TAILLE DE CLIENT (Q1 = 12 plus petits ... Q4 = 12 plus grands) — dBrier x1e3 moyen par quartile\n" + "=" * 160)
SIZEQ = pd.qcut(pd.Series(NTR).rank(method="first"), 4, labels=False).values
show = ["compat_k2", PROPOSED, "fomo_upd", "fedper", "fb_fedper_t", "fb_fedper_raw", "ditto_l0.03", "fb_ditto_t"]
print(pd.DataFrame({r: [f"{dB[r][SIZEQ == q].mean()*1e3:+.3f}" for q in range(4)] for r in show}, index=[f"Q{q+1}" for q in range(4)]).to_string())

print("\n" + "=" * 160 + "\nD. VERDICT (criteres fixes avant le run) — la granularite 'paire x couche' apporte-t-elle quelque chose vs un fallback au niveau MODELE ?\n" + "=" * 160)
print("  Comparateurs du verdict : fb_fedper_t et fb_fedper_raw (FedPer est le persistant le plus precis). Ditto est donne en information.")
print("  Le gate est 'moins precis' si l'IC95 de mean(gate - X) est < 0. Il est 'plus sur' si l'IC95 de harm_mat(gate - X) est < 0 OU si l'IC95 de dCVaR10 est > 0.")
less_acc, safer = {}, {}
for b in ["fb_fedper_t", "fb_fedper_raw", "fb_ditto_t", "fb_ditto_raw"]:
    c = contrast(PROPOSED, b); less_acc[b] = c["mean"][2] < 0; safer[b] = (c["hmat"][2] < 0) or (c["dcvar"][1] > 0)
    tag = "VERDICT" if "fedper" in b else "info   "
    print(f"  [{tag}] vs {b:<14} moins precis: {str(less_acc[b]):<5} | plus sur: {str(safer[b]):<5} | mean {fm(c['mean'])} | harm_mat {fm(c['hmat'],1,3)} | dCVaR10 {fm(c['dcvar'])}")
key = ["fb_fedper_t", "fb_fedper_raw"]
if any(less_acc[b] for b in key):
    print("  => FALLBACK PLUS PRECIS : un choix au niveau modele est significativement plus precis que le gate. Positionnement a revoir avant l'Introduction.")
elif all(safer[b] for b in key):
    print("  => GRANULARITE ETABLIE (vs FedPer) : a precision non inferieure, le gate est plus sur / a une meilleure queue qu'un fallback au niveau modele.")
else:
    print("  => GRANULARITE NON ETABLIE : le gate n'apporte pas de securite demontree au-dela d'un fallback au niveau modele. "
          "Positionner la contribution sur l'abstention validee, et presenter le niveau paire x couche comme un choix de conception, pas comme un gain demontre.")
print(f"\nTotal time: {(time.time() - t00) / 60:.1f} min")
