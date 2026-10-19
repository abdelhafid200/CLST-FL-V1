# Construction d'une fédération hétérogène (48 clients) à partir d'eICU, version 3, sans clusters.
# Changements par rapport à v2 : un séjour par patient, prétraitement global, vrai décalage de concept
# (compatibilité graduée via un mélange de concepts de base), pickles qui ne contiennent que des arrays.
import os, json, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# Config
DATA_PATH    = "../dataset/"
PATIENT_PATH = f"{DATA_PATH}/patient.csv.gz"
APS_PATH     = f"{DATA_PATH}/apacheApsVar.csv.gz"
APACHE_PATH  = f"{DATA_PATH}/apachePatientResult.csv.gz"
BASE_OUTPUT_DIR = "data_v3"

K = 48
MISSING_STRATEGY   = "impute"      # "impute" (garde tout) ou "drop" (cas complets, peu de données)
MIN_CLIENT_RECORDS = 700
MIN_CLIENT_CLASS   = 40            # min. positifs et négatifs par client
MAX_TRIES = 50
FLOOR     = 0.05                   # plancher uniforme d'affectation (0.25 diluait trop)
M_BASE    = 8                      # nb de concepts de base (ce ne sont pas des clusters de clients)
PI_ALPHA  = (0.05, 2.0)            # Dirichlet par client : α_k log-uniforme (spécialistes <-> généralistes)
                                   # mettre (0.5, 0.5) pour une compatibilité strictement graduée
CONCEPT_NORM = 1.6                 # norme fixe de la fonction de concept de chaque client

LEVELS = {                         # gx: covariables | gy: prévalence | sq: taille | ca: décalage de concept
    "iid":    dict(gx=0.0, gy=0.0, sq=0.0, ca=0.0),   # contrôle : plancher de bruit
    "low":    dict(gx=0.5, gy=0.3, sq=0.3, ca=0.3),
    "medium": dict(gx=1.0, gy=0.6, sq=0.5, ca=0.6),
    "high":   dict(gx=1.5, gy=1.0, sq=0.8, ca=1.0),   # ca=1 : étiquettes entièrement synthétiques
}
TEST_SIZE, VAL_SIZE, SEED = 0.20, 0.20, 42
FEATURE_COLS = ["pao2", "fio2", "creatinine", "bun", "wbc", "temperature",
                "meanbp", "eyes", "motor", "verbal", "vent", "dialysis"]
ID_COL, Y_COL = "patientunitstayid", "mortality"
np.random.seed(SEED); os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 1) Chargement + un sejour par patient
# colonnes de la table patient à vérifier : uniquepid, patienthealthsystemstayid, unitvisitnumber
patient = pd.read_csv(PATIENT_PATH, usecols=[ID_COL, "hospitalid", "uniquepid",
                      "patienthealthsystemstayid", "unitvisitnumber"]).drop_duplicates(ID_COL)
aps = pd.read_csv(APS_PATH, usecols=[ID_COL] + FEATURE_COLS).drop_duplicates(ID_COL)
header = pd.read_csv(APACHE_PATH, nrows=0).columns
a_cols = [ID_COL, "actualhospitalmortality"] + (["apacheversion"] if "apacheversion" in header else [])
apache = pd.read_csv(APACHE_PATH, usecols=a_cols)
raw = apache["actualhospitalmortality"].str.strip().str.upper()
apache[Y_COL] = raw.map({"ALIVE": 0, "EXPIRED": 1})
apache = apache.dropna(subset=[Y_COL])
nl = apache.groupby(ID_COL)[Y_COL].nunique()
apache = apache[~apache[ID_COL].isin(nl[nl > 1].index)]
if "apacheversion" in apache.columns:
    apache["_p"] = (apache["apacheversion"].astype(str).str.strip().str.upper() != "IVA").astype(int)
    apache = apache.sort_values([ID_COL, "_p"], kind="mergesort")
apache = apache.drop_duplicates(ID_COL)[[ID_COL, Y_COL]]
apache[Y_COL] = apache[Y_COL].astype(int)

df = (patient.merge(aps, on=ID_COL, how="inner", validate="one_to_one")
             .merge(apache, on=ID_COL, how="inner", validate="one_to_one"))
n_stays = len(df)
df = (df.sort_values(["uniquepid", "patienthealthsystemstayid", "unitvisitnumber"])
        .drop_duplicates("uniquepid").reset_index(drop=True))     # un seul séjour par patient
assert df[ID_COL].is_unique and df["uniquepid"].is_unique
print(f"Séjours fusionnés : {n_stays:,} -> un séjour par patient : {len(df):,} | mortalité {df[Y_COL].mean():.2%}")

# 2) Valeurs manquantes (-1 = manquant)
df[FEATURE_COLS] = df[FEATURE_COLS].replace(-1, np.nan)
for c in FEATURE_COLS:
    print(f"  {c:<12} manquants : {int(df[c].isna().sum()):>8,}")
if MISSING_STRATEGY == "drop":
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"Conservés : {len(df):,} | mortalité {df[Y_COL].mean():.2%} | hôpitaux sources {df['hospitalid'].nunique()}")

X_raw  = df[FEATURE_COLS].values.astype(float)      # peut contenir des NaN (imputés globalement plus bas)
y_real = df[Y_COL].values.astype(int)
ids, hosp = df[ID_COL].values, df["hospitalid"].values
n, d = len(df), len(FEATURE_COLS)

# copie standardisée sans NaN : sert uniquement à concevoir la partition et les étiquettes
X_fill = np.where(np.isnan(X_raw), np.nanmedian(X_raw, axis=0), X_raw)
Xs = (X_fill - X_fill.mean(0)) / (X_fill.std(0) + 1e-8)

# 3) Parametres clients
def draw_params(seed):
    rng = np.random.default_rng(seed)
    zq  = rng.normal(size=K)                                                      # effet de taille
    U   = rng.normal(size=(K, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)  # profil de covariables
    lam = np.clip(rng.normal(size=K), -1.5, 1.5)                                  # effet de prévalence
    Wm  = rng.normal(0, CONCEPT_NORM / np.sqrt(d), size=(M_BASE, d))              # concepts de base
    bm  = rng.normal(-2.0, 0.6, size=M_BASE)
    a_k = np.exp(rng.uniform(np.log(PI_ALPHA[0]), np.log(PI_ALPHA[1]), size=K))
    Pi  = np.vstack([rng.dirichlet(a * np.ones(M_BASE)) for a in a_k])            # mélange par client
    Wc  = Pi @ Wm
    Wc  = CONCEPT_NORM * Wc / np.linalg.norm(Wc, axis=1, keepdims=True)           # renormalisation
    return dict(zq=zq, U=U, lam=lam, Wc=Wc, bc=Pi @ bm, Pi=Pi, a_k=a_k, u=rng.random(n))

def assign(P, gx, gy, sq):
    logits = sq * P["zq"][None, :] + gx * (Xs @ P["U"].T) + gy * (y_real[:, None] * P["lam"][None, :])
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    p = (1 - FLOOR) * p + FLOOR / K
    return np.clip((p.cumsum(1) < P["u"][:, None]).sum(1), 0, K - 1)

def relabel(cid, P, ca, seed):
    if ca <= 0:
        return y_real.copy()
    s = 1 / (1 + np.exp(-(np.einsum("nd,nd->n", Xs, P["Wc"][cid]) + P["bc"][cid])))
    p = (1 - ca) * y_real + ca * s
    return (np.random.default_rng(seed + 999).random(n) < p).astype(int)

# 4) Diagnostics
def pairwise_auroc(clients):
    fits = [LogisticRegression(max_iter=1000).fit(c["X_train"], c["y_train"]) for c in clients.values()]
    M = np.full((K, K), np.nan)
    for j in range(K):
        for k in range(K):
            try:
                M[j, k] = roc_auc_score(clients[k]["y_val"], fits[j].decision_function(clients[k]["X_val"]))
            except ValueError:
                pass
    return M

def headroom(clients):
    """Marge de personnalisation : modèle poolé vs local vs poolé + ajustement local (sur la VALIDATION)."""
    Xp = np.vstack([c["X_train"] for c in clients.values()])
    yp = np.concatenate([c["y_train"] for c in clients.values()])
    pooled = LogisticRegression(max_iter=1000).fit(Xp, yp)
    r = []
    for c in clients.values():
        lt, lv = pooled.decision_function(c["X_train"]), pooled.decision_function(c["X_val"])
        pp = pooled.predict_proba(c["X_val"])[:, 1]
        pl = LogisticRegression(max_iter=1000).fit(c["X_train"], c["y_train"]).predict_proba(c["X_val"])[:, 1]
        ps = LogisticRegression(max_iter=1000).fit(np.c_[c["X_train"], lt], c["y_train"]).predict_proba(np.c_[c["X_val"], lv])[:, 1]
        r.append([brier_score_loss(c["y_val"], q) for q in (pp, pl, ps)] +
                 [roc_auc_score(c["y_val"], q) for q in (pp, pl, ps)])
    r = np.array(r)
    print(f"  Brier   poolé {r[:,0].mean():.4f} | local {r[:,1].mean():.4f} | poolé+local {r[:,2].mean():.4f}")
    print(f"  AUROC   poolé {r[:,3].mean():.4f} | local {r[:,4].mean():.4f} | poolé+local {r[:,5].mean():.4f}")
    print(f"  marge de personnalisation : gain Brier relatif {100*(1-r[:,2].mean()/r[:,0].mean()):.2f}% | "
          f"clients où poolé+local bat poolé : {100*np.mean(r[:,2] < r[:,0]):.0f}%")
    return r

# 5) Construction par niveau
for lvl, hp in LEVELS.items():
    print("\n" + "#" * 70 + f"\n#  NIVEAU = {lvl}   {hp}\n" + "#" * 70)
    for t in range(MAX_TRIES):
        P = draw_params(SEED + t)
        cid = assign(P, hp["gx"], hp["gy"], hp["sq"])
        y_c = relabel(cid, P, hp["ca"], SEED + t)
        sizes = np.bincount(cid, minlength=K)
        pos = np.bincount(cid, weights=y_c, minlength=K).astype(int)
        if (sizes.min() >= MIN_CLIENT_RECORDS and pos.min() >= MIN_CLIENT_CLASS
                and (sizes - pos).min() >= MIN_CLIENT_CLASS):
            break
    else:
        raise RuntimeError(f"[{lvl}] contraintes non satisfaites (taille min {sizes.min()}, positifs min {pos.min()})")
    print(f"  ✓ paramètres tirés à la tentative {t}")

    # découpe brute par client, puis prétraitement global ajusté sur l'union des train
    rawsp = {}
    for k in range(K):
        idx = np.where(cid == k)[0]
        Xk, yk, ik = X_raw[idx], y_c[idx], ids[idx]
        Xtv, Xte, ytv, yte, itv, ite = train_test_split(Xk, yk, ik, test_size=TEST_SIZE, random_state=SEED, stratify=yk)
        Xtr, Xva, ytr, yva, itr, iva = train_test_split(Xtv, ytv, itv, test_size=VAL_SIZE, random_state=SEED, stratify=ytv)
        rawsp[k] = (Xtr, ytr, itr, Xva, yva, iva, Xte, yte, ite)
    Xtr_all = np.vstack([rawsp[k][0] for k in range(K)])
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr_all)
    sc  = StandardScaler().fit(imp.transform(Xtr_all))
    prep = lambda A: sc.transform(imp.transform(A))

    clients = {}
    for k in range(K):
        Xtr, ytr, itr, Xva, yva, iva, Xte, yte, ite = rawsp[k]
        Xtr_imp = imp.transform(Xtr)
        clients[k] = {
            "client_id": k, "phenotype": f"client_{k:02d}", "scaler": None,       # plus d'objet sklearn
            "X_train": prep(Xtr), "y_train": ytr, "X_val": prep(Xva), "y_val": yva,
            "X_test": prep(Xte), "y_test": yte,
            "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
            "n_deaths_train": int(ytr.sum()), "mortality_rate_train": float(ytr.mean()),
            "mortality_rate_val": float(yva.mean()), "mortality_rate_test": float(yte.mean()),
            "raw_mean": Xtr_imp.mean(0), "raw_scale": Xtr_imp.std(0) + 1e-8,      # moments bruts du client
            "ids_train": itr, "ids_val": iva, "ids_test": ite,
            "n_source_hospitals": int(len(np.unique(hosp[np.where(cid == k)[0]]))),
        }
    all_ids = np.concatenate([c[s] for c in clients.values() for s in ("ids_train", "ids_val", "ids_test")])
    assert len(all_ids) == len(np.unique(all_ids)), "FUITE : un séjour apparaît plusieurs fois"

    nn = np.array([c["n_train"] + c["n_val"] + c["n_test"] for c in clients.values()])
    pr = np.array([c["mortality_rate_train"] for c in clients.values()])
    print(f"  ✓ {K} clients | {len(all_ids):,} séjours uniques")
    print(f"  taille     min/méd/max : {nn.min():,} / {int(np.median(nn)):,} / {nn.max():,}")
    print(f"  prévalence min/méd/max : {pr.min():.3f} / {np.median(pr):.3f} / {pr.max():.3f}")
    M = pairwise_auroc(clients); off = ~np.eye(K, dtype=bool)
    asym = np.nanmean([abs(M[a, b] - M[b, a]) for a in range(K) for b in range(a + 1, K)])
    print(f"  AUROC intra-client {np.nanmean(np.diag(M)):.3f} | inter-client {np.nanmean(M[off]):.3f} | "
          f"asymétrie moyenne {asym:.3f}")
    hr = headroom(clients)

    Wn = P["Wc"] / np.linalg.norm(P["Wc"], axis=1, keepdims=True); TRUE_COMPAT = Wn @ Wn.T
    out_dir = os.path.join(BASE_OUTPUT_DIR, f"clients_hetero_{lvl}"); os.makedirs(out_dir, exist_ok=True)
    for k, c in clients.items():
        with open(os.path.join(out_dir, f"client_{k:02d}.pkl"), "wb") as f:
            pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"n_clients": K, "input_dim": d, "feature_cols": FEATURE_COLS, "level": lvl, "hetero_params": hp,
            "missing_strategy": MISSING_STRATEGY, "seed": SEED, "attempt": int(t),
            "n_unique_stays": int(len(all_ids)), "n_source_hospitals": int(len(np.unique(hosp))),
            "global_mean": sc.mean_.tolist(), "global_scale": sc.scale_.tolist(),
            "true_compat": TRUE_COMPAT.tolist(),          # pour analyse/oracle uniquement, jamais comme entrée
            "pi": P["Pi"].tolist(), "alpha_k": P["a_k"].tolist(),
            "phenotypes": sorted(c["phenotype"] for c in clients.values()),
            "client_summary": {k: {s: c[s] for s in ("n_train", "n_val", "n_test", "mortality_rate_train",
                                                      "mortality_rate_val", "mortality_rate_test",
                                                      "n_source_hospitals")} for k, c in clients.items()}}
    dump = lambda o: o.item() if hasattr(o, "item") else str(o)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=dump)
    with open(os.path.join(out_dir, "diagnostic.json"), "w") as f:
        json.dump({"within": float(np.nanmean(np.diag(M))), "cross": float(np.nanmean(M[off])),
                   "asymmetry": float(asym), "headroom_rows": hr.tolist()}, f, indent=2)
    print(f"  ✓ sauvegardé dans '{out_dir}/'")
print("\n✅ Terminé.")