# CLST-FL — code for "validation-gated peer transfer with abstention" (eICU, 48 clients)

Scripts used to produce every number in the manuscript. Each script is a **single self-contained notebook cell**
(model, FedAvg trajectory, rules, bootstrap, tables). Scripts 01–07 **write no file** — the printed output is the only
result; archive it (see `outputs/` below). Script 00 writes one CSV to `results_confirm/`.

## Requirements
Python 3, `numpy`, `pandas`, `scipy`, `torch` (CPU works, GPU used if available). See `requirements.txt`
(exact versions were not recorded — TODO fill in). Runs are deterministic for fixed seeds *within the same environment*.

## Data (not included)
All scripts read `BASE_DIR/clients_hetero_{LEVEL}/` (`client_*.pkl` for the 48 clients and `metadata.json`) with
`LEVEL ∈ {iid, low, medium, high}`. Default `BASE_DIR` is the local folder `data_v3` (written by `data_00_build_clients_v3.py`) —
adapt it if needed (line "`BASE_DIR = ... # ... à adapter`").
The files are derived from the eICU Collaborative Research Database (PhysioNet, credentialed access).
`scripts/data_00_build_clients_v3.py` builds `data_v3` from the raw eICU tables (`patient`, `apacheApsVar`, `apachePatientResult`; edit `DATA_PATH`, default `../dataset/`). It writes `data_v3/clients_hetero_{iid,low,medium,high}/` (48 `client_XX.pkl` + `metadata.json` + `diagnostic.json`). Level parameters (g_x,g_y,s_q,c_a): iid (0,0,0,0), low (0.5,0.3,0.3,0.3), medium (1.0,0.6,0.5,0.6), high (1.5,1.0,0.8,1.0); split seed 42. Run it first. NB: clients are a synthetic partition of pooled patients (not hospitals), and labels are partly (c_a=0.3/0.6) or fully (c_a=1, high) synthetic.

## Natural (hospital) partition — heterogeneity check
`scripts/data_01_natural_hospital_partition.py` takes the 48 largest eICU hospitals as clients with real labels (same features, split, preprocessing and diagnostics as `data_00_...`), and prints, with identical definitions for the four generated levels (read from `data_v3/`), n_train, prevalence, intra/inter AUROC, personalization headroom, domain-classifier AUROC and dispersion of log SMR. It supports the statement in the manuscript about natural versus generated heterogeneity. With `SAVE_CLIENTS=True` it writes `clients_hetero_natural/` (no oracle) that experiment scripts can read with `LEVEL="natural"`.

`scripts/data_02_measure_non_iid_real_vs_artificial.py` (output in `outputs/data_02_measure_non_iid_output.txt`) measures non-IID-ness with identical definitions on the four generated levels, the 48 largest real hospitals and the 48 largest hospital units (reads `data_v4_reel/cohorte_eicu_reel.csv.gz` from step 4a and `data_v3/`); it reproduces the generated-level numbers exactly. Source of manuscript Table "noniid".

## Scripts, seeds and manuscript tables
| Script | Header (first comment) | Seeds | Level(s) | ~Time/seed | Manuscript |
|---|---|---|---|---|---|
| `00_exp_exploration_K8.py` | RUN CONFIRMATOIRE (K=8, règles fedavg_global … sel@t) | 123–127 | iid, then high (run twice, edit `LEVEL`) | ~10 min | Exploration with K=8 (justifies the K=2 deviation) |
| `01_exp_next_attribution.py` | EXP-Next : Controlled Selectivity / Gate Attribution | 123–127 | high | ~9.5 min | Attribution phase (text of §5.1) |
| `02_exp_final_confirmation.py` | EXP-Final : Gate Attribution CONFIRMATOIRE | 223–227 | high | ~8.8 min | Tables 3 and 4 (primary result, P1–P5) |
| `03_exp_levels.py` | EXP-Levels | 223–227 | iid, low, medium (auto) | ~9 min/level | Tables 5 and 6 |
| `04_exp_ablation.py` | EXP-Ablation | 323–327 | high | ~6.8 min | Table 7 |
| `05_exp_baselines_v1.py` | EXP-Baselines (Ditto λ∈{0.1,1.0}, no early stopping) | 423–425 | high | ~44–54 min | First baseline run (FedPer +0.943 vs gate +0.193, cited in text) |
| `06_exp_baselines_v2.py` | EXP-Baselines-2 (λ grid, early stopping on validation) | 423–425 | high | ~46 min | Table 8 |
| `08_exp_natural_partition.py` | EXP-Natural : le gate sur la partition naturelle (48 hôpitaux, vraies étiquettes; `LEVEL="natural"`, nécessite `data_01` avec `SAVE_CLIENTS=True`) | 623–627 | natural | ≤9 min | Table `natural` (output in `outputs/08_exp_natural_output.txt`) |
| `07_exp_fallback.py` | EXP-Fallback | 523–525 | high | ~26–28 min | Table 9 |

Run order is free (scripts are independent); use the order above. Fixed settings: MLP 12→64→32→1 (2,945 params),
50 rounds, 5 local epochs, Adam 1e-3, batch 64, dropout 0.10, γ=0.1, K=2 (K=8 only in 00), snapshots at rounds 8,13,…,48,
τ ∈ {1.645, 2.0}, harm threshold δ=1e-4, bootstrap B=2000 over clients.

## Outputs to archive
Copy the printed output of each run into `outputs/` (e.g. `outputs/02_exp_final_high.txt`). These texts are the source of all
manuscript numbers. TODO: add them (they are your uploaded run logs).

## Metric note
In scripts 01-04 "harm_frac" is the per-client share of accepted pairs that are harmful (0 if the client abstains), averaged over clients — a transfer-level rate, not the share of clients worse than FT. The latter ("materially worse", >0.1e-3) is reported by scripts 05-07 and in section E of script 08.

## Known caveats (disclosed in the manuscript)
- Seeds 123–127 were used twice: first by script 00 (K=8), then by script 01 (K=2).
- Script for the exploration seeds 42–44 is not included (not available in this archive) — TODO.
- τ=1.645/2.0 are operational thresholds, not α-levels; no sequential validity.
- Baselines are our own "-style" reimplementations.
