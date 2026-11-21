# CWT–BiGRU lithofacies imputation: leakage-free evaluation

Code for the paper *Continuous Wavelet Transform and Bidirectional Gated Recurrent Unit Framework for Lithofacies Imputation in Partly Cored Wells: Leakage-Free Single-Well and Blind-Well Evaluation* (Khalaf and Al-Mudhafar, SPE Journal, revised manuscript SJ-0926-0136).

Every number, table and figure in the paper is produced by the scripts below. The JSON files in `results/` are the outputs used in the paper.

## Data

* **Case 2 (Council Grove Group, Hugoton and Panoma fields, Kansas).** Clone the SEG 2016 machine-learning contest repository and place its folder at `data/seg2016`:
  `git clone https://github.com/seg/2016-ml-contest data/seg2016`
  The scripts use `facies_vectors.csv`, `nofacies_data.csv` and `blind_stuart_crawford_core_facies.csv` from that repository. The data are not redistributed here.
* **Case 1 (single well).** Place the file at `data/karpur.csv` (819 rows; columns include depth, caliper, gamma ray, bulk density, neutron porosity and the core facies). It is available from the corresponding author on request.

## Environment

Python 3.10 or later; `pip install -r requirements.txt`. All runs were made on a CPU. Set `LFI_THREADS=1 OMP_NUM_THREADS=1` when running several scripts at once.

## Order of runs

```
# How the earlier accuracy arose (Fig. 5)
python3 reproduce_original.py data/karpur.csv 5 results/reproduce_original.json

# Classification of uncored intervals
python3 karpur.py data/karpur.csv 16 3 results/karpur_b16.json                    # Case 1, Table 3
python3 hugoton.py data/seg2016 blind 5 results/hug_blind.json                    # Case 2, Table 4
python3 run_hug_complex.py data/seg2016 5 results/hug_blind_complex.json          # Case 2, framework with Re/Im CWT
python3 tune_hug.py data/seg2016 results/tune_hug.json rfctx,raw_bigru:p1,hyb_bigru:mag:p1,hyb_bigru:complex:p1,raw_bigru:p2,hyb_bigru:mag:p2,hyb_bigru:complex:p2   # Table S-4
python3 sens.py results/sens_cwt.json                                             # Table 6

# Imputation of gaps (Table 5, Figs. 6-9)
OFF0=1 python3 impute2.py karpur 8,32,2,16 1 2 results/imp_final_karpur.json
OFF0=1 python3 impute2.py karpur 4 1 2 results/imp_final_karpur_G4.json
python3 impute3.py hugoton 8 1 3 results/imp_full_hugoton_G8.json
python3 impute3.py hugoton 32,2,16,4 1 3 results/imp_full_hugoton_rest.json
python3 curves.py results/curves_hug8.json                                        # Fig. 10

# Numbers and figures
python3 summarize.py                                                              # results/paper_numbers.json
python3 fig_method.py figs; python3 fig_results.py figs; python3 fig_results2.py figs
```

## What the scripts do

* `core.py` — CWT features (complex Morlet `cmor1.5-1.0`, 16 scales from 2 to 64 samples, 64-sample reflect padding; modulus or real and imaginary parts), normalization fitted on training samples only, the sequence networks and their training loop with early stopping on validation data.
* `geo_metrics.py` — SCI (Eq. 5), bed fragmentation ratio, transition entropy (Eq. 6) and thin-bed recall.
* `karpur.py` — blocked cross-validation in the single well (8-ft blocks, 4-sample guard, five folds, three repeats).
* `hugoton.py` — training on nine wells and testing on the blind wells STUART and CRAWFORD; leave-one-well-out mode for design selection.
* `impute2.py`, `impute3.py` — gap imputation (gaps of G ft every 5G ft, five rotated layouts), nearest-facies copy, random forest with bounding facies, the network ablations, McNemar inputs and the permutation (depth-shift) importance.
* `dev/` — earlier development versions of the imputation and single-well scripts, kept for transparency; they are not used for any reported number.

No test label is used for training, normalization, early stopping or model selection in any script, and depth is not an input to any model.
