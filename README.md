# CLST-FL — code for "validation-gated peer transfer with abstention" (eICU, 48 clients)

Scripts used to run the experiments of the manuscript. Each script is a single self-contained notebook cell
(model, FedAvg trajectory, rules, bootstrap, tables). The experiment scripts 01–08 write no file: the printed output is
the result. Script 00 also writes one CSV to `results_confirm/`, and script 02b writes its two export files.

## Requirements
Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` and `torch` (see `requirements.txt`).
The code runs on CPU; a GPU is used if available. Package versions are not pinned. For fixed seeds, runs are
deterministic within the same software environment.

## Data
**No patient data is included in this repository.**
All data are derived from the eICU Collaborative Research Database, available on PhysioNet under credentialed access:
https://physionet.org/content/eicu-crd/

> Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. The eICU Collaborative Research Database, a freely
> available multi-center database for critical care research. *Scientific Data*. 2018;5:180178.

### Generated partitions (`data_00`)
`scripts/data_00_build_clients_v3.py` builds the client files from the raw eICU tables `patient`, `apacheApsVar` and
`apachePatientResult` (set `DATA_PATH`, default `../dataset/`). It writes `data_v3/clients_hetero_{iid,low,medium,high}/`
(48 `client_XX.pkl`, `metadata.json`, `diagnostic.json`). Run it first.

- One ICU stay per patient, 12 APACHE variables, per-client split 64/16/20 (train/validation/test, seed 42),
  imputation and standardisation fitted on the union of the training sets.
- Level parameters (g_x, g_y, s_q, c_a): iid (0, 0, 0, 0), low (0.5, 0.3, 0.3, 0.3), medium (1.0, 0.6, 0.5, 0.6),
  high (1.5, 1.0, 0.8, 1.0).
- Clients are a synthetic partition of pooled patients (not hospitals). Labels are partly synthetic for low and
  medium (c_a = 0.3, 0.6) and fully synthetic for high (c_a = 1).

All experiment scripts read `BASE_DIR/clients_hetero_{LEVEL}/`. The default `BASE_DIR` is the local folder `data_v3`;
change the `BASE_DIR = ...` line if your data is elsewhere.

### Natural (hospital) partition (`data_01`)
`scripts/data_01_natural_hospital_partition.py` uses the 48 largest eICU hospitals as clients, with real labels and the
same variables, split and preprocessing as `data_00`. It prints heterogeneity diagnostics (n_train, prevalence,
intra/inter-client AUROC, personalisation headroom, domain-classifier AUROC, dispersion of log SMR) for this partition
and, with the same definitions, for the four generated levels. With `SAVE_CLIENTS=True` it writes
`data_v3/clients_hetero_natural/` (no oracle compatibility), which script 08 reads with `LEVEL="natural"`.

### Non-IID measurement (`data_02`)
`scripts/data_02_measure_non_iid_real_vs_artificial.py` measures non-IID-ness with identical definitions on the four
generated levels, the 48 largest real hospitals and the 48 largest hospital units. It reads `data_v3/` and a cohort
file `data_v4_reel/cohorte_eicu_reel.csv.gz`, produced by a separate cohort-extraction step that is not included in this
repository. Output: `outputs/data_02_measure_non_iid_output.txt`.

## Scripts, seeds and manuscript tables
| Script | Description | Seeds | Level(s) | ~Time/seed | Manuscript |
|---|---|---|---|---|---|
| `00_exp_exploration_K8.py` | Run with K=8 candidate sources, all rules | 123–127 | iid, then high (run twice, edit `LEVEL`) | ~10 min | K = 8 run (Section 4.8, Table 3) |
| `01_exp_next_attribution.py` | Gate attribution with K=2 | 123–127 | high | ~9.5 min | Attribution phase (Section 5.1) |
| `02_exp_final_confirmation.py` | Confirmatory run, pre-specified criteria P1–P5 | 223–227 | high | ~8.8 min | Tables 4, 5 and 7; Figs. 3–5; high row of Table 6 |
| `02b_exp_final_export.py` | Same as 02 (same calculations and seeds), plus export of per-client and per-pair data (`exp02_fig_data.npz`, `exp02_pairs.csv`) | 223–227 | high | ~8.8 min | Fig. 4 and Section 5.7 (Fig. 7) |
| `03_exp_levels.py` | Same rules on the other levels | 223–227 | all levels in `BASE_DIR` except high (iid, low, medium) | ~9 min/level | Table 6 (iid, low, medium); Fig. 6 |
| `04_exp_ablation.py` | Ablation of the gate statistic at equal budget | 323–327 | high | ~6.8 min | Table 8 |
| `05_exp_baselines_v1.py` | First baseline run (Ditto λ ∈ {0.1, 1.0}, no early stopping) | 423–425 | high | ~44–54 min | Earlier baseline run (Section 5.8) |
| `06_exp_baselines_v2.py` | Tuned baselines (λ grid, early stopping on validation) | 423–425 | high | ~46 min | Table 9; Fig. 8 |
| `07_exp_fallback.py` | Model-level fallback baselines | 523–525 | high | ~26–28 min | Table 10; Fig. 8 |
| `08_exp_natural_partition.py` | Gate on the natural hospital partition (real labels; needs `data_01` with `SAVE_CLIENTS=True`) | 623–627 | natural | ≤9 min | Tables 11 and 12; Fig. 6 |

The data scripts give the heterogeneity diagnostics: `data_00` and `data_01` for Table 1 (the FT Brier column comes
from the experiment scripts), `data_02` for Table 2 and Fig. 2.

The scripts are independent and can be run in any order. Fixed settings: MLP 12→64→32→1 (2,945 parameters),
50 rounds, 5 local epochs, Adam (lr 1e-3), batch 64, dropout 0.10, γ = 0.1, K = 2 (K = 8 only in script 00),
snapshots at rounds 8, 13, …, 48, τ ∈ {1.645, 2.0}, harm threshold δ = 1e-4, bootstrap B = 2000 over clients.
Selection and gating use the validation split only; the test split is used only for evaluation.

## Outputs
`outputs/` contains the printed output of scripts 05 (`05_exp_baselines_v1_output.txt`), 06
(`06_exp_baselines_v2_output.txt`), 08 (`08_exp_natural_output.txt`) and `data_02` (`data_02_measure_non_iid_output.txt`).
The other results are obtained by running the scripts.

## Metric note
In scripts 01–04 and 08, "harm_frac" is the per-client share of accepted pairs that are harmful (0 if the client
abstains), averaged over clients: a transfer-level rate, not the share of clients worse than FT. The share of clients
materially worse than FT (> 0.1e-3) is reported by scripts 05–07 and in section E of script 08.

## Known caveats
- Seeds 123–127 were used twice: first by script 00 (K = 8), then by script 01 (K = 2).
- The script of the preliminary exploration (seeds 42–43, Appendix A) is not included.
- τ = 1.645 / 2.0 are operational thresholds, not α-levels; the gate has no sequential validity guarantee.
- Baselines (FedPer, Ditto, FedFomo) are our own reimplementations.
