# Partition naturelle par hôpital (les 48 plus grands hôpitaux eICU, vraies étiquettes) et diagnostics d'hétérogénéité.
# Même chargement, mêmes 12 variables, même découpage (64/16/20, seed 42), même prétraitement global et mêmes diagnostics
# (AUROC intra/inter, marge de personnalisation) que data_00_build_clients_v3.py.
# Sorties :
#   (1) tableau : n_train, prévalence, AUROC intra/inter, marge de personnalisation, pour "hospitals (real)"
#   (2) deux métriques complémentaires, calculées avec la même définition pour les 4 niveaux générés (si data_v3/ existe) et pour la partition naturelle :
#         - AUROC du classifieur de domaine (régression logistique multinomiale client ~ 12 variables ; AUROC OvR macro, jeu de test)
#         - dispersion du log SMR (écart-type sur les clients de log(décès observés / décès attendus par le modèle poolé), jeu de test)
#   (3) option SAVE_CLIENTS : écrit data_v3/clients_hetero_natural/ au même format (utilisable par les scripts 02/03/... avec LEVEL="natural"
#       ; pas de true_compat -> pas de règle oracle)
# Aucun entraînement fédéré ici : seulement des régressions logistiques (quelques minutes).
import os, json, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# Config (identique à data_00_build_clients_v3.py)
DATA_PATH    = "../dataset/"
PATIENT_PATH = f"{DATA_PATH}/patient.csv.gz"
APS_PATH     = f"{DATA_PATH}/apacheApsVar.csv.gz"
APACHE_PATH  = f"{DATA_PATH}/apachePatientResult.csv.gz"
BASE_OUTPUT_DIR = "data_v3"
K = 48
MIN_CLIENT_RECORDS = 700           # mêmes contraintes que les niveaux générés
MIN_CLIENT_CLASS   = 40
TEST_SIZE, VAL_SIZE, SEED = 0.20, 0.20, 42
FEATURE_COLS = ["pao2", "fio2", "creatinine", "bun", "wbc", "temperature",
                "meanbp", "eyes", "motor", "verbal", "vent", "dialysis"]
ID_COL, Y_COL = "patientunitstayid", "mortality"
SAVE_CLIENTS = False               # True : écrit data_v3/clients_hetero_natural/
COMPARE_LEVELS = ["iid", "low", "medium", "high"]   # relus depuis BASE_OUTPUT_DIR s'ils existent
np.random.seed(SEED)

# 1) Chargement + un sejour par patient (copie exacte de v3)
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
        .drop_duplicates("uniquepid").reset_index(drop=True))
assert df[ID_COL].is_unique and df["uniquepid"].is_unique
print(f"Séjours fusionnés : {n_stays:,} -> un séjour par patient : {len(df):,} | mortalité {df[Y_COL].mean():.2%}")
print("  (contrôle : doit donner 148,532 -> 119,881 | 9.49% comme dans la sortie de data_00_build_clients_v3.py)")
df[FEATURE_COLS] = df[FEATURE_COLS].replace(-1, np.nan)
print(f"Hôpitaux sources : {df['hospitalid'].nunique()}")

# 2) Les 48 plus grands hopitaux (contraintes de v3)
g = df.groupby("hospitalid")[Y_COL].agg(n="size", pos="sum").sort_values("n", ascending=False)
g["neg"] = g["n"] - g["pos"]
ok = g[(g["n"] >= MIN_CLIENT_RECORDS) & (g["pos"] >= MIN_CLIENT_CLASS) & (g["neg"] >= MIN_CLIENT_CLASS)]
print(f"Hôpitaux éligibles (>= {MIN_CLIENT_RECORDS} patients et >= {MIN_CLIENT_CLASS} par classe) : {len(ok)}")
if len(ok) < K:
    raise RuntimeError(f"seulement {len(ok)} hôpitaux éligibles pour K={K} : baisser MIN_CLIENT_RECORDS ou K")
chosen = list(ok.index[:K])
print(f"48 plus grands hôpitaux retenus : taille min/méd/max = {int(ok['n'].iloc[:K].min()):,} / "
      f"{int(ok['n'].iloc[:K].median()):,} / {int(ok['n'].iloc[:K].max()):,} patients "
      f"({int(ok['n'].iloc[:K].sum()):,} patients au total, {100*ok['n'].iloc[:K].sum()/len(df):.1f}% de la cohorte)")
hmap = {h: k for k, h in enumerate(chosen)}
sel = df["hospitalid"].isin(hmap).values
X_raw = df.loc[sel, FEATURE_COLS].values.astype(float)
y_all = df.loc[sel, Y_COL].values.astype(int)
ids   = df.loc[sel, ID_COL].values
cid   = df.loc[sel, "hospitalid"].map(hmap).values
d = len(FEATURE_COLS)

# 3) Decoupe + pretraitement global (copie exacte de v3)
rawsp = {}
for k in range(K):
    idx = np.where(cid == k)[0]
    Xk, yk, ik = X_raw[idx], y_all[idx], ids[idx]
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
        "client_id": k, "phenotype": f"client_{k:02d}", "scaler": None,
        "X_train": prep(Xtr), "y_train": ytr, "X_val": prep(Xva), "y_val": yva,
        "X_test": prep(Xte), "y_test": yte,
        "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
        "n_deaths_train": int(ytr.sum()), "mortality_rate_train": float(ytr.mean()),
        "mortality_rate_val": float(yva.mean()), "mortality_rate_test": float(yte.mean()),
        "raw_mean": Xtr_imp.mean(0), "raw_scale": Xtr_imp.std(0) + 1e-8,
        "ids_train": itr, "ids_val": iva, "ids_test": ite,
        "n_source_hospitals": 1, "hospitalid": int(chosen[k]),
    }
all_ids = np.concatenate([c[s] for c in clients.values() for s in ("ids_train", "ids_val", "ids_test")])
assert len(all_ids) == len(np.unique(all_ids)), "FUITE : un séjour apparaît plusieurs fois"

# 4) Diagnostics (copie des fonctions de v3 + métriques complémentaires)
def pairwise_auroc(cl):
    ks = sorted(cl)
    fits = [LogisticRegression(max_iter=1000).fit(cl[k]["X_train"], cl[k]["y_train"]) for k in ks]
    M = np.full((len(ks), len(ks)), np.nan)
    for a, j in enumerate(ks):
        for b, k in enumerate(ks):
            try: M[a, b] = roc_auc_score(cl[k]["y_val"], fits[a].decision_function(cl[k]["X_val"]))
            except ValueError: pass
    return M

def headroom(cl):
    Xp = np.vstack([c["X_train"] for c in cl.values()]); yp = np.concatenate([c["y_train"] for c in cl.values()])
    pooled = LogisticRegression(max_iter=1000).fit(Xp, yp); r = []
    for c in cl.values():
        lt, lv = pooled.decision_function(c["X_train"]), pooled.decision_function(c["X_val"])
        pp = pooled.predict_proba(c["X_val"])[:, 1]
        pl = LogisticRegression(max_iter=1000).fit(c["X_train"], c["y_train"]).predict_proba(c["X_val"])[:, 1]
        ps = LogisticRegression(max_iter=1000).fit(np.c_[c["X_train"], lt], c["y_train"]).predict_proba(np.c_[c["X_val"], lv])[:, 1]
        r.append([brier_score_loss(c["y_val"], q) for q in (pp, pl, ps)] + [roc_auc_score(c["y_val"], q) for q in (pp, pl, ps)])
    return np.array(r)

def domain_auc_and_smr(cl):
    """AUROC OvR macro d'un classifieur de domaine (client ~ variables) et dispersion du log SMR (modèle poolé), sur le TEST."""
    ks = sorted(cl)
    Xtr = np.vstack([cl[k]["X_train"] for k in ks]); ctr = np.concatenate([[k] * len(cl[k]["y_train"]) for k in ks])
    Xte = np.vstack([cl[k]["X_test"] for k in ks]);  cte = np.concatenate([[k] * len(cl[k]["y_test"]) for k in ks])
    dom = LogisticRegression(max_iter=500).fit(Xtr, ctr)
    P = dom.predict_proba(Xte)
    dauc = roc_auc_score(cte, P, multi_class="ovr", average="macro", labels=dom.classes_)
    yp = np.concatenate([cl[k]["y_train"] for k in ks])
    pooled = LogisticRegression(max_iter=1000).fit(Xtr, yp)
    lsmr = []
    for k in ks:
        exp = pooled.predict_proba(cl[k]["X_test"])[:, 1].sum(); obs = cl[k]["y_test"].sum()
        if obs > 0 and exp > 0: lsmr.append(np.log(obs / exp))
    return float(dauc), float(np.std(lsmr, ddof=1))

def summarize(name, cl):
    nt = np.array([c["n_train"] for c in cl.values()]); pr = np.array([c["mortality_rate_train"] for c in cl.values()])
    M = pairwise_auroc(cl); off = ~np.eye(len(cl), dtype=bool); hr = headroom(cl)
    dauc, sd_lsmr = domain_auc_and_smr(cl)
    row = dict(level=name, n_train=f"{nt.min()} / {int(np.median(nt))} / {nt.max()}",
               prevalence=f"{pr.min():.3f} / {np.median(pr):.3f} / {pr.max():.3f}",
               auroc_intra_inter=f"{np.nanmean(np.diag(M)):.3f} / {np.nanmean(M[off]):.3f}",
               headroom_pct=round(100 * (1 - hr[:, 2].mean() / hr[:, 0].mean()), 2),
               pct_clients_pooled_local_beats_pooled=round(100 * np.mean(hr[:, 2] < hr[:, 0])),
               domain_auroc=round(dauc, 3), sd_log_smr=round(sd_lsmr, 3))
    return row

rows = [summarize("hospitals (real)", clients)]
for lvl in COMPARE_LEVELS:                       # même définition des métriques pour les niveaux générés
    cdir = os.path.join(BASE_OUTPUT_DIR, f"clients_hetero_{lvl}")
    if not os.path.isdir(cdir): print(f"[info] {cdir} absent : niveau {lvl} non recalculé"); continue
    cl = {}
    for f in sorted(os.listdir(cdir)):
        if f.startswith("client_") and f.endswith(".pkl"):
            with open(os.path.join(cdir, f), "rb") as fh: c = pickle.load(fh)
            cl[c["client_id"]] = c
    rows.append(summarize(lvl, cl))
res = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("\n" + "=" * 100 + "\nHETEROGENEITE : partition naturelle (hôpitaux) vs niveaux générés (mêmes définitions)\n" + "=" * 100)
print(res.to_string(index=False))
print("\nLecture : headroom_pct = gain Brier relatif (validation) de poolé+ajustement local vs poolé (négatif = le poolé suffit) ; "
      "sd_log_smr inclut le bruit d'échantillonnage (plancher = niveau iid).")

# 5) Sauvegarde optionnelle
if SAVE_CLIENTS:
    out_dir = os.path.join(BASE_OUTPUT_DIR, "clients_hetero_natural"); os.makedirs(out_dir, exist_ok=True)
    for k, c in clients.items():
        with open(os.path.join(out_dir, f"client_{k:02d}.pkl"), "wb") as f: pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"n_clients": K, "input_dim": d, "feature_cols": FEATURE_COLS, "level": "natural",
            "hetero_params": {"natural_partition": "48 largest hospitals, real labels"}, "seed": SEED,
            "n_unique_stays": int(len(all_ids)), "hospital_ids": [int(h) for h in chosen],
            "global_mean": sc.mean_.tolist(), "global_scale": sc.scale_.tolist()}   # pas de true_compat : pas d'oracle
    with open(os.path.join(out_dir, "metadata.json"), "w") as f: json.dump(meta, f, indent=2)
    print(f"✓ sauvegardé dans '{out_dir}/'  (utiliser LEVEL=\"natural\" dans les scripts d'expérience)")
print("\n✅ Terminé.")
