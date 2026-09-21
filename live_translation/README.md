# English–Arabic live speech translation benchmark

This folder contains a **full-corpus GPU experiment** on MuST-C English speech to Arabic text. The default configuration trains on every official training segment and generates translations for every segment in both official test splits. It uses a frozen [SeamlessM4T v2 large](https://huggingface.co/facebook/seamless-m4t-v2-large) speech model with trainable decoder adapters. The synthetic CPU run described at the end is only a software check; it is not the benchmark.

Clone the repository first, then run the remaining commands from its root, with `live_translation/` as a child folder. Downloaded data, speech encodings, and results are written under ignored `live_translation/data/` and `live_translation/runs/` paths.

## 1. Install and check CUDA

Python 3.10 is the tested baseline. Install the dependencies and verify that PyTorch sees a GPU. If the installed PyTorch wheel does not support your driver, reinstall it using the command for your machine from the [official PyTorch selector](https://pytorch.org/get-started/locally/):

```bash
git clone https://github.com/alirezaabdollahpour/Causal_DRO.git
cd Causal_DRO
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r live_translation/requirements.txt
python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch and a visible GPU are required"; print(torch.__version__, torch.cuda.get_device_name(0))'
```

The check must print the GPU name before starting the benchmark. The full run needs substantial disk space for the corpus, Hugging Face model, cached speech states, checkpoints, and results, plus enough GPU memory for SeamlessM4T v2 large. The full configuration has no small-dataset cap.

## 2. Download all MuST-C En–Ar data and prepare full manifests

The source is [MuST-C En–Ar on Kaggle](https://www.kaggle.com/datasets/sebaeymohamed/must-c-en-ar), dataset version 1. The downloader requests **all** files in that release, resumes incomplete files, checks expected sizes, and records local SHA-256 hashes. It may take a long time and can be rerun after interruption.

```bash
python live_translation/scripts/download_mustc.py \
  --root live_translation/data/kaggle_mustc_en_ar --workers 12
python live_translation/scripts/prepare_mustc.py \
  --root live_translation/data/kaggle_mustc_en_ar \
  --output live_translation/data/kaggle_mustc_en_ar/manifests \
  --validate-audio
python live_translation/scripts/prepare_full_mustc.py \
  --source-dir live_translation/data/kaggle_mustc_en_ar/manifests \
  --output-dir live_translation/data/kaggle_mustc_en_ar/full_manifests
```

The first preparation script also writes optional sampled manifests for quick checks. The benchmark uses **only** `full_manifests`. The second preparation script verifies official row counts, manifest hashes, split separation, audio paths, and audio headers. Each row pairs a timed span of 16 kHz speech with its English transcript and Arabic reference. No duration, talk, text, or score filter is applied to the full manifests.

| Official split | Segments | Use |
| --- | ---: | --- |
| `train` | 212,085 | Every row is used in each default training epoch; also the continuation donor bank. |
| `dev` | 1,073 | Selects learned evaluation attacker checkpoints. |
| `tst-COMMON` | 2,019 | Final inference and scoring. |
| `tst-HE` | 578 | Final inference and scoring. |

The two test splits are combined into a 2,597-row evaluation manifest while keeping each row's official split label. Neither test split supplies training examples or continuation donors.

## 3. Fetch the pinned model and run the full benchmark

The configuration pins `facebook/seamless-m4t-v2-large` to revision `5f8cc790b19fc3f67a61c105133b20b34e3dcb76`. Transformers fetches it automatically on first use. To download it before the long job:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/seamless-m4t-v2-large",
    revision="5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
)
PY
```

Run one seed through cache preparation, full training, and full test inference. The three commands use the same output directory and may be resumed after interruption:

```bash
python -m live_translation.benchmark \
  --config live_translation/configs/full_ar.json \
  --stage prepare --output live_translation/runs/full_ar/seed_11
python -m live_translation.benchmark \
  --config live_translation/configs/full_ar.json \
  --stage train --output live_translation/runs/full_ar/seed_11
python -m live_translation.benchmark \
  --config live_translation/configs/full_ar.json \
  --stage evaluate --output live_translation/runs/full_ar/seed_11
```

Or run the same stages in one command. The launcher checks for a visible CUDA GPU before starting:

```bash
bash live_translation/scripts/run_benchmark.sh \
  --config live_translation/configs/full_ar.json \
  --stage all --output live_translation/runs/full_ar/seed_11
```

The default `warmup_epochs=1`, `defender_epochs=1`, and `eval_fit_epochs=1` make a complete shuffled pass over all 212,085 training rows for clean warmup, **each** defender, and **each** learned evaluation attacker fit. The encoded cache includes all official train, dev, and test rows; the donor bank aliases all training rows. Evaluation writes a translation and score for each of the 2,597 test rows for every configured defender, attack, and `wait_k` condition. The full Arabic configuration has five defenders and five evaluation attacks. Use `full_ar_rnn.json` to add the causal RNN defender and attack.

For three independent seeds, use distinct result directories. The same encoded cache can be reused when the model and encoding settings are unchanged:

```bash
for seed in 11 22 33; do
  bash live_translation/scripts/run_benchmark.sh \
    --config live_translation/configs/full_ar.json \
    --stage all --seed "$seed" \
    --output "live_translation/runs/full_ar/seed_$seed"
done
```

`--stage prepare` encodes the complete corpus, `train` saves warmup and defender checkpoints, and `evaluate` fits evaluation attackers on training data, selects them on dev, and scores the complete test split. A new output directory needs training before evaluation. After preparing manifests, `python -m live_translation.benchmark --config live_translation/configs/full_ar.json --check` checks local paths and configuration; it does not run the model or verify the remote model ID.

## Change the experiment while retaining full-data coverage

Use a new `--output` path for each setting. These command-line options override the JSON configuration:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--seed` | `11` | Random seed. |
| `--lambda` | `3` | Penalty weight in cross entropy minus `lambda × transport_cost`. |
| `--warmup-epochs` | `1` | Complete training passes for clean adapter warmup. |
| `--defender-epochs` | `1` | Complete training passes for each defender; the outer training count. |
| `--eval-fit-epochs` | `1` | Complete training passes for each learned evaluation attacker. |
| `--attack-inner-steps` | `1` | Attacker optimizer updates within each defender update. |
| `--picnn-solver-steps` | `8` | PICNN inner solve iterations during training and inference. |
| `--train-duchi-steps` | `2` | Direct Duchi iterations during defender training. |
| `--eval-duchi-steps` | `3` | Direct anticipative Duchi iterations at inference. |
| `--eval-adaptive-causal-duchi-steps` | `4` | Direct adaptive causal Duchi iterations at inference. |

For example, this remains a full-corpus run with two passes for each training phase:

```bash
bash live_translation/scripts/run_benchmark.sh \
  --config live_translation/configs/full_ar.json \
  --stage all --output live_translation/runs/full_ar/lam2_seed7_two_epochs \
  --lambda 2 --seed 7 \
  --warmup-epochs 2 --defender-epochs 2 --eval-fit-epochs 2 \
  --attack-inner-steps 3 --picnn-solver-steps 12 \
  --train-duchi-steps 3 --eval-duchi-steps 5 \
  --eval-adaptive-causal-duchi-steps 6
```

The `--warmup-steps`, `--defender-steps`, and `--eval-fit-steps` options set fixed optimizer update counts instead of epoch counts. In the full-corpus configuration, counts shorter than one complete pass are rejected. Use the epoch options for complete passes. Other parameters, including batch sizes, learning rates, attack architecture, and method lists, can be changed in a copied JSON configuration.

| Method | Role and information available | Method-specific controls |
| --- | --- | --- |
| `nominal` / `clean` | Train the adapter or evaluate it on unchanged speech states. | `warmup_epochs`, `defender_epochs`, `defender_lr` |
| `causal_picnn` | Learned attack using the observed speech prefix and previously chosen states. | `attack_lr`, `attack_inner_steps`, `picnn_width`, `picnn_depth`, `picnn_solver_steps`, `picnn_initialization` |
| `anticipative_picnn` | Learned attack using the complete source utterance, without its reference. | Same PICNN controls. |
| `causal_rnn` | GRU attack using observed states and a fixed time coordinate; available in `full_ar_rnn.json`. | `rnn_hidden_dim`, `rnn_head_dim`, `rnn_time_scale`, `attack_lr`, `attack_inner_steps` |
| `adaptive_causal_duchi` | Optimize each action over continuations from the training donor bank, then commit it. | `conditional_samples`, `conditional_shortlist`, `adaptive_causal_duchi_restarts`, `duchi_max_evaluations`, `train_duchi_steps`, `eval_adaptive_causal_duchi_steps` |
| `anticipative_duchi` | Optimize a complete path using the scored reference; a label-informed diagnostic. | `train_duchi_steps`, `eval_duchi_steps`, `duchi_step_size`, `attack_tolerance` |

| JSON field | Full configuration default | Meaning |
| --- | ---: | --- |
| `batch_size`, `eval_batch_size` | `4`, `1` | Training and evaluation batch sizes. Training rows longer than `training_singleton_seconds=20` are processed alone. |
| `defender_lr`, `attack_lr` | `2e-4`, `1e-6` | Adam learning rates for adapter and learned attackers. |
| `adapter_rank` | `4` | Decoder adapter rank. |
| `chunk_ms`, `encoder_context_ms`, `wait_k` | `640`, `20480`, `[5]` | Speech block duration, rolling observed audio window, and initial source block count. |
| `picnn_width`, `picnn_depth` | `64`, `3` | Convex attacker size. |
| `rnn_hidden_dim`, `rnn_head_dim`, `rnn_time_scale` | `64`, `64`, `128` | Causal RNN size and time coordinate in `full_ar_rnn.json`. |
| `conditional_samples`, `conditional_shortlist`, `adaptive_causal_duchi_restarts` | `2`, `128`, `2` | Training-donor samples, approximate donor shortlist, and causal restart count. |
| `duchi_max_evaluations`, `attack_tolerance` | `10000`, `1e-4` | Direct attack evaluation budget and stopping tolerance. |
| `cost_scale`, `training_radius_rule`, `evaluation_scaling` | `1`, `none`, `none` | Fixed transport cost denominator; no radius scaling. |
| `eval_selection`, `eval_selection_every_epochs` | `development_payoff`, `1` | Pick a learned evaluation attacker by dev payoff after each full pass. |
| `max_new_tokens`, `evaluation_max_seconds` | `512`, `100` | Generation cap and declared evaluation audio horizon. |

The learned attackers use encoded speech states. `causal_picnn` and `causal_rnn` act from observed prefixes; `anticipative_picnn` sees the complete source speech but no translation reference. `adaptive_causal_duchi` optimizes each committed action over training-donor continuations. `anticipative_duchi` uses the complete scored reference and is a label-informed diagnostic. Attacks perturb encoded speech states, not waveforms. Their objective is mean translation cross entropy minus `lambda × transport_cost`, where cost is the sum of squared valid state changes divided by `cost_scale`. Finite optimization budgets produce attack candidates, not certified worst cases.

## Results and verification

Each result directory contains the resolved `config.json`, `run_identity.json`, execution trace, checkpoints, and `conditions/` JSON files. Each condition file records the expected and evaluated test population, per-split summaries, and utterance-level hypotheses, references, cross entropy, cost, completion status, and timing. Check the `population` field in every condition file: it should report `evaluated=2597`, with `tst-COMMON=2019` and `tst-HE=578`. Run identity checks protect resume against changes in code, model assets, and encoded data.

The complete full-corpus GPU grid has not been executed as part of preparing this repository, so this README reports the configured protocol and coverage checks rather than benchmark scores. Model and dataset usage terms are on their linked source pages.

## Optional synthetic CPU check

This check needs no dataset or GPU. It exercises the package on generated short feature sequences and a small decoder; its scores do not measure speech translation quality:

```bash
python -m pytest live_translation/tests -q
python -m live_translation \
  --config live_translation/configs/toy.yaml \
  --output live_translation/runs/toy_check
```

The folder also contains an older En–De/fairseq pilot in `backend.py`, `data.py`, and `simuleval_agent.py`. It requires a separate MuST-C En–De release, official fairseq checkpoint, and compatible fairseq/SimulEval environment.
