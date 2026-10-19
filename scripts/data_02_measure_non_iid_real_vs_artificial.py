# Mesure du non-IID : vrais hopitaux vs clients artificiels (v3)
#
# Script independant, a lancer apres l'etape 4a (il lit
# data_v4_reel/cohorte_eicu_reel.csv.gz) et la ou se trouve data_v3/
# (clients_hetero_iid, _low, _medium, _high). CPU, ~5 min.
#
# Memes mesures, memes 12 variables, memes regles de decoupe que le
# pretraitement v3, appliquees a 6 partitions :
#   v3 iid / low / medium / high  : clients artificiels (etiquettes synthetiques des que ca > 0)
#   reel : hopitaux               : client = un vrai hopital, vraie mortalite
#   reel : unites                 : client = une unite de soins d'un hopital
#                                   (ex. USI cardiaque, USI medicale), vraie mortalite
#
# 5 familles de mesures (plus grand = plus heterogene) :
#   quantite         : coefficient de variation des tailles de clients
#   etiquettes       : ecart moyen de mortalite au global ; min / mediane / max
#   covariables      : AUROC d'un classifieur "ce client vs les autres" (0.5 = IID)
#   concept          : SMR = deces observes / deces attendus par un modele poole
#                      (1 = IID) ; dispersion du SMR entre clients = ecart-type de
#                      log(SMR) (0 = IID) ; chute d'AUROC quand on applique le
#                      modele d'un autre client
#   personnalisation : gain de Brier d'un ajustement local sur le modele poole
#                      (c'est cette grandeur qui decide si FedAvg + FT sert a quelque chose)
#
# mesurer_non_iid(clients, nom) prend le meme dictionnaire que le pretraitement v3,
# on peut donc aussi l'appeler a la fin de celui-ci.

import os, glob, json, pickle, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 0. configuration (identique au pretraitement v3)
V3_DIR = os.environ.get("NONIID_V3_DIR", "data_v3")
COHORT_PATH = os.environ.get("NONIID_COHORT", os.path.join("data_v4_reel", "cohorte_eicu_reel.csv.gz"))
OUT_DIR = os.environ.get("NONIID_OUT_DIR", "mesure_non_iid")
os.makedirs(OUT_DIR, exist_ok=True)

FEATS_V3 = ["pao2", "fio2", "creatinine", "bun", "wbc", "temperature",
            "meanbp", "eyes", "motor", "verbal", "vent", "dialysis"]
Y_COL = "mortality"
K_MAX = 48                                   # meme nombre de clients que v3 (les plus grands)
MIN_RECORDS, MIN_CLASS = 700, 40             # memes contraintes que v3
TEST_SIZE, VAL_SIZE, SEED = 0.20, 0.20, 42
V3_LEVELS = ["iid", "low", "medium", "high"]
# valeurs affichees par le pretraitement v3 (controle de reproduction)
REF_V3 = {"iid": (0.795, 0.795, -1.86, 15), "low": (0.732, 0.686, 2.12, 69),
          "medium": (0.738, 0.610, 8.36, 96), "high": (0.755, 0.539, 16.15, 98)}


def fit_lr(X, y):
    return LogisticRegression(max_iter=1000).fit(X, y)


# 1. la mesure (reutilisable telle quelle dans le pretraitement v3)
def mesurer_non_iid(clients, nom, seed=SEED):
    """clients : {id: {"X_train", "y_train", "X_val", "y_val", ("y_test")}} deja standardises."""
    C = [clients[k] for k in sorted(clients)]
    K = len(C)
    ytv = [np.concatenate([c["y_train"], c["y_val"]]).astype(float) for c in C]
    sizes = np.array([len(c["y_train"]) + len(c["y_val"]) + len(c.get("y_test", [])) for c in C])
    prev = np.array([y.mean() for y in ytv])
    p_glob = np.concatenate(ytv).mean()

    # Covariables : classifieur "ce client vs les autres"
    rng = np.random.default_rng(seed)
    Xtr_all = [c["X_train"] for c in C]
    Xva_all = [c["X_val"] for c in C]
    dom = []
    for i, c in enumerate(C):
        o_tr = np.vstack([Xtr_all[j] for j in range(K) if j != i])
        o_va = np.vstack([Xva_all[j] for j in range(K) if j != i])
        ntr, nva = len(c["X_train"]), len(c["X_val"])
        a = o_tr[rng.choice(len(o_tr), min(ntr, len(o_tr)), replace=False)]
        b = o_va[rng.choice(len(o_va), min(nva, len(o_va)), replace=False)]
        m = fit_lr(np.vstack([c["X_train"], a]), np.r_[np.ones(ntr), np.zeros(len(a))])
        dom.append(roc_auc_score(np.r_[np.ones(nva), np.zeros(len(b))], m.decision_function(np.vstack([c["X_val"], b]))))
    dom = np.array(dom)

    # Concept (a) : transfert croise, comme pairwise_auroc de v3
    fits = [fit_lr(c["X_train"], c["y_train"]) for c in C]
    M = np.full((K, K), np.nan)
    for j in range(K):
        for k in range(K):
            try:
                M[j, k] = roc_auc_score(C[k]["y_val"], fits[j].decision_function(C[k]["X_val"]))
            except ValueError:
                pass
    off = ~np.eye(K, dtype=bool)
    intra, inter = float(np.nanmean(np.diag(M))), float(np.nanmean(M[off]))

    # Concept (b) SMR et personnalisation, comme headroom de v3
    pooled = fit_lr(np.vstack(Xtr_all), np.concatenate([c["y_train"] for c in C]))
    r, smr, smr_z = [], [], []
    for c in C:
        lt, lv = pooled.decision_function(c["X_train"]), pooled.decision_function(c["X_val"])
        pp = pooled.predict_proba(c["X_val"])[:, 1]
        pl = fit_lr(c["X_train"], c["y_train"]).predict_proba(c["X_val"])[:, 1]
        ps = fit_lr(np.c_[c["X_train"], lt], c["y_train"]).predict_proba(np.c_[c["X_val"], lv])[:, 1]
        r.append([brier_score_loss(c["y_val"], q) for q in (pp, pl, ps)] +
                 [roc_auc_score(c["y_val"], q) for q in (pp, pl, ps)])
        e = pooled.predict_proba(np.vstack([c["X_train"], c["X_val"]]))[:, 1]
        o = np.concatenate([c["y_train"], c["y_val"]]).sum()
        smr.append(o / e.sum())
        smr_z.append((o - e.sum()) / np.sqrt((e * (1 - e)).sum()))
    r, smr, smr_z = np.array(r), np.array(smr), np.array(smr_z)

    return dict(partition=nom, K=K, taille_med=int(np.median(sizes)), taille_cv=float(sizes.std() / sizes.mean()),
                mort_min=float(prev.min()), mort_med=float(np.median(prev)), mort_max=float(prev.max()),
                ecart_mort=float(np.abs(prev - p_glob).mean()),
                auc_domaine=float(dom.mean()),
                smr_min=float(smr.min()), smr_med=float(np.median(smr)), smr_max=float(smr.max()),
                smr_disp=float(np.std(np.log(np.clip(smr, 1e-6, None)))),
                part_smr_sig=float(np.mean(np.abs(smr_z) > 2)),
                auroc_intra=intra, auroc_inter=inter, chute_transfert=intra - inter,
                brier_poole=float(r[:, 0].mean()), brier_local=float(r[:, 1].mean()),
                brier_poole_local=float(r[:, 2].mean()),
                auroc_poole=float(r[:, 3].mean()), auroc_poole_local=float(r[:, 5].mean()),
                gain_perso_pct=float(100 * (1 - r[:, 2].mean() / r[:, 0].mean())),
                part_perso=float(np.mean(r[:, 2] < r[:, 0])))


# 2. partitions v3 (clients artificiels deja construits)
def load_v3(folder):
    out = {}
    for f in sorted(os.listdir(folder)):
        if f.startswith("client_") and f.endswith(".pkl"):
            with open(os.path.join(folder, f), "rb") as fh:
                c = pickle.load(fh)
            out[c["client_id"]] = c
    return out


def find_v3(level):
    cands = [os.path.join(V3_DIR, f"clients_hetero_{level}")]
    return next((c for c in cands if os.path.exists(os.path.join(c, "metadata.json"))), None)


# 3. partitions reelles (meme decoupe et meme pretraitement global que v3)
def build_real(coh, key_cols):
    st = coh.groupby(key_cols)[Y_COL].agg(["size", "sum"])
    st = st[(st["size"] >= MIN_RECORDS) & (st["sum"] >= MIN_CLASS) & (st["size"] - st["sum"] >= MIN_CLASS)]
    st = st.sort_values("size", ascending=False).head(K_MAX)
    keys = coh[key_cols].apply(tuple, axis=1) if len(key_cols) > 1 else coh[key_cols[0]]
    rawsp = {}
    for k, key in enumerate(st.index):
        sub = coh[keys == key]
        Xk, yk = sub[FEATS_V3].to_numpy(float), sub[Y_COL].to_numpy(int)
        Xtv, Xte, ytv, yte = train_test_split(Xk, yk, test_size=TEST_SIZE, random_state=SEED, stratify=yk)
        Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=VAL_SIZE, random_state=SEED, stratify=ytv)
        rawsp[k] = (Xtr, ytr, Xva, yva, Xte, yte)
    if len(rawsp) < 4:
        return None, st
    Xtr_all = np.vstack([v[0] for v in rawsp.values()])
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr_all)
    sc = StandardScaler().fit(imp.transform(Xtr_all))
    prep = lambda A: sc.transform(imp.transform(A))
    clients = {k: dict(X_train=prep(v[0]), y_train=v[1], X_val=prep(v[2]), y_val=v[3], X_test=prep(v[4]), y_test=v[5])
               for k, v in rawsp.items()}
    return clients, st


# 4. calcul
t0 = time.time()
results = []
for lvl in V3_LEVELS:
    d = find_v3(lvl)
    if d is None:
        print(f"[INFO] v3 '{lvl}' introuvable dans {V3_DIR} -> ignore (renseigner NONIID_V3_DIR)")
        continue
    results.append(mesurer_non_iid(load_v3(d), f"v3 {lvl}"))
    print(f"  v3 {lvl} mesure ({time.time() - t0:.0f}s)")

assert os.path.exists(COHORT_PATH), f"Cohorte introuvable : {COHORT_PATH}. Lancer d'abord l'etape 4a (ou renseigner NONIID_COHORT)."
coh = pd.read_csv(COHORT_PATH, low_memory=False)
missing_feats = [c for c in FEATS_V3 if c not in coh.columns]
assert not missing_feats, f"variables v3 absentes de la cohorte : {missing_feats}"
partitions = [("reel : hopitaux", ["hospitalid"])]
if "unittype" in coh.columns:
    coh["unittype"] = coh["unittype"].fillna("(inconnu)")
    partitions.append(("reel : unites", ["hospitalid", "unittype"]))
real_info = {}
for nom, keys in partitions:
    clients, st = build_real(coh, keys)
    real_info[nom] = st
    if clients is None:
        print(f"[INFO] {nom} : moins de 4 clients eligibles -> ignore")
        continue
    results.append(mesurer_non_iid(clients, nom))
    print(f"  {nom} mesure : {len(clients)} clients ({time.time() - t0:.0f}s)")
    if nom == "reel : unites":
        print("     types d'unites retenus : " + ", ".join(f"{k} ({v})" for k, v in
              st.reset_index()["unittype"].value_counts().items()))

res = pd.DataFrame(results)
res.to_csv(os.path.join(OUT_DIR, "mesure_non_iid.csv"), index=False)

# 5. affichage
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("\n" + "=" * 120)
print("1) MESURES DU NON-IID (memes 12 variables, meme decoupe que v3)")
print("=" * 120)
blocks = [("QUANTITE et ETIQUETTES", ["partition", "K", "taille_med", "taille_cv", "mort_min", "mort_med", "mort_max", "ecart_mort"]),
          ("COVARIABLES et CONCEPT", ["partition", "auc_domaine", "smr_min", "smr_med", "smr_max", "smr_disp",
                                     "part_smr_sig", "auroc_intra", "auroc_inter", "chute_transfert"]),
          ("PERSONNALISATION", ["partition", "brier_poole", "brier_local", "brier_poole_local", "auroc_poole",
                                "auroc_poole_local", "gain_perso_pct", "part_perso"])]
for title, cols in blocks:
    print(f"\n-- {title} --")
    print(res[cols].round(3).to_string(index=False))

print("\n" + "=" * 120)
print("2) CONTROLE : les niveaux v3 recalcules doivent redonner les chiffres de votre pretraitement")
print("=" * 120)
for lvl, (i_ref, x_ref, g_ref, s_ref) in REF_V3.items():
    row = res[res.partition == f"v3 {lvl}"]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    ok = abs(r.auroc_intra - i_ref) < 0.002 and abs(r.auroc_inter - x_ref) < 0.002 and abs(r.gain_perso_pct - g_ref) < 0.05
    print(f"  v3 {lvl:6s} intra {r.auroc_intra:.3f} (ref {i_ref}) | inter {r.auroc_inter:.3f} (ref {x_ref}) | "
          f"gain perso {r.gain_perso_pct:.2f} % (ref {g_ref}) | clients {100 * r.part_perso:.0f} % (ref {s_ref}) -> "
          f"{'OK' if ok else 'ECART (dossier v3 different de celui du run imprime ?)'}")

# Lecture : a quel niveau v3 ressemble chaque partition reelle ?
print("\n" + "=" * 120)
print("LECTURE : niveau v3 le plus proche, mesure par mesure")
print("=" * 120)
KEY = [("auc_domaine", "covariables"), ("ecart_mort", "etiquettes"), ("smr_disp", "concept (SMR)"),
       ("chute_transfert", "concept (transfert)"), ("gain_perso_pct", "personnalisation")]
v3r = res[res.partition.str.startswith("v3")]
for _, r in res[res.partition.str.startswith("reel")].iterrows():
    if len(v3r) == 0:
        break
    near = []
    for col, lab_ in KEY:
        best = v3r.iloc[(v3r[col] - r[col]).abs().argmin()]
        near.append(best.partition.replace("v3 ", ""))
        print(f"  {r.partition:16s} {lab_:20s} : {r[col]:8.3f}  -> le plus proche : {best.partition} ({best[col]:.3f})")
    vc = pd.Series(near).value_counts()
    if vc.iloc[0] > len(near) / 2:
        print(f"  => {r.partition} ressemble surtout a v3 '{vc.index[0]}' ({vc.iloc[0]}/{len(near)} mesures)\n")
    else:
        print(f"  => {r.partition} : profil MIXTE ({', '.join(f'{k} x{v}' for k, v in vc.items())}) : "
              "heterogene sur certains axes, pas sur d'autres\n")
print("NOTE : part_smr_sig (tableau) grandit avec la taille des clients ; la lecture utilise smr_disp, moins sensible a la taille.")
print("RAPPEL : pour la porte de redeploiement, l'heterogeneite ENTRE clients compte moins que le changement")
print("DANS un client au cours du temps (etape 4a). Elle decide surtout si FedAvg + FT produit des candidats differents.")

# 6. figure (4 mesures, une par panneau)
SYN, REAL = "#2a78d6", "#eb6834"
panels = [("auc_domaine", "Covariables : AUROC client vs autres\n(0.5 = aucune difference)", 0.5),
          ("ecart_mort", "Etiquettes : ecart moyen de mortalite\nau taux global", 0.0),
          ("smr_disp", "Concept : dispersion du SMR entre clients\n(ecart-type de log(observes / attendus))", 0.0),
          ("gain_perso_pct", "Personnalisation : gain de Brier (%)\nd'un ajustement local sur le modele poole", 0.0)]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
names = res.partition.tolist()
colors = [SYN if n.startswith("v3") else REAL for n in names]
for ax, (col, title, base) in zip(axes.ravel(), panels):
    vals = res[col].to_numpy()
    xs = np.arange(len(vals))
    ax.bar(xs, vals - base, bottom=base, width=0.62, color=colors)
    span = np.nanmax(np.abs(vals - base)) if len(vals) else 1
    for x_, v in zip(xs, vals):
        ax.text(x_, v + (0.03 * span if v >= base else -0.03 * span), f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}",
                ha="center", va="bottom" if v >= base else "top", fontsize=9, color="#333333")
    ax.axhline(base, color="#888888", linewidth=0.9)
    ax.set_xticks(xs); ax.set_xticklabels([n.replace("reel : ", "reel\n").replace("v3 ", "v3\n") for n in names], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(min(base, np.nanmin(vals)) - 0.12 * span, max(base, np.nanmax(vals)) + 0.15 * span)
    ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=SYN), plt.Rectangle((0, 0), 1, 1, color=REAL)]
fig.legend(handles, ["clients artificiels (v3)", "clients reels (vraie mortalite)"], loc="upper center", ncol=2,
           frameon=False, fontsize=10)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig(os.path.join(OUT_DIR, "mesure_non_iid.png"), dpi=150)
plt.show()

print(f"\nSauvegarde dans '{OUT_DIR}/' : mesure_non_iid.csv, mesure_non_iid.png  (total {time.time() - t0:.0f}s)")
print("A me renvoyer : la sortie console (sections 1, 2, LECTURE) et la figure mesure_non_iid.png.")