# Live translation experiments

This folder contains a small CPU audit and a speech translation benchmark. The audit runs on generated data and is useful for checking the training and attack code. The benchmark uses English speech and Arabic text from MuST-C, with a frozen SeamlessM4T v2 speech model and trainable decoder adapters. Run the commands below from the repository root, with `live_translation/` as a child folder.

Attacks change the model's **encoded speech states**, not the audio waveform. The main objective for an attack is mean translation cross entropy minus `lam × transport_cost`, where `transport_cost` is the sum of squared changes to valid state blocks divided by `cost_scale`. Translation is generated without access to the reference. The `anticipative_duchi` diagnostic is the exception: it uses the complete scored reference when finding its perturbation.

## Install and try the CPU audit

Python 3.10 is the tested baseline. Install a PyTorch build suitable for your machine, then the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
# Install a matching PyTorch build using https://pytorch.org/get-started/locally/
python -m pip install -r live_translation/requirements.txt
python -m pytest live_translation/tests -q
python -m live_translation \
  --config live_translation/configs/toy.yaml \
  --output live_translation/runs/toy_example
```

The toy run needs no download or GPU. It uses short generated feature sequences and a small decoder. Its scores check execution; they do not measure speech translation quality. Set a new `--output` path for each changed configuration. A completed run checks its saved configuration before resuming.

For example, this changes the penalty, random seed, and optimizer work in the audit:

```bash
python -m live_translation \
  --config live_translation/configs/toy.yaml \
  --output live_translation/runs/toy_lam2_seed7 \
  --lambda 2 --seed 7 --warmup-steps 20 --defender-steps 10 \
  --attack-inner-steps 3 --attack-fit-steps 20 \
  --attack-steps 20 --picnn-solver-steps 12
```

`--warmup-steps` is clean adapter training, `--defender-steps` is training for each defender, and `--attack-inner-steps` is attacker updates per defender update. `--attack-fit-steps` fits an evaluation attacker against a frozen defender; `--attack-steps` controls direct attack optimization in the toy runner. `--picnn-solver-steps` controls the learned attack's inner conjugate solve.

## Download the English–Arabic data and model

The speech benchmark uses [MuST-C En–Ar on Kaggle](https://www.kaggle.com/datasets/sebaeymohamed/must-c-en-ar), dataset version 1. It preserves the official train, dev, `tst-COMMON`, and `tst-HE` splits. The release contains about 64.6 GB of extracted files, including 2,462 talk recordings. The downloader resumes individual files, checks their expected sizes, and saves local SHA-256 records. Downloaded data, generated manifests, caches, and run results belong in the ignored `live_translation/data/` and `live_translation/runs/` directories; they are not included in Git.

```bash
python live_translation/scripts/download_mustc.py \
  --root live_translation/data/kaggle_mustc_en_ar --metadata-only --workers 8
python live_translation/scripts/download_mustc.py \
  --root live_translation/data/kaggle_mustc_en_ar --audio-only --workers 12
python live_translation/scripts/prepare_mustc.py \
  --root live_translation/data/kaggle_mustc_en_ar \
  --output live_translation/data/kaggle_mustc_en_ar/manifests \
  --validate-audio
python live_translation/scripts/prepare_full_mustc.py \
  --source-dir live_translation/data/kaggle_mustc_en_ar/manifests \
  --output-dir live_translation/data/kaggle_mustc_en_ar/full_manifests
```

The first two commands may also run as one unfiltered download command. Each manifest row pairs a timed span of 16 kHz talk audio with its English transcript and Arabic reference, using the supplied YAML offset and duration. The preparation scripts check text/YAML row alignment, split and talk identity, and audio headers. The full manifest keeps every official row: 212,085 train segments, 1,073 dev segments, 2,019 `tst-COMMON` segments, and 578 `tst-HE` segments. Both test splits are combined only for evaluation, retaining their split labels. The full training split also supplies the continuation donors for the causal direct attack. No held-out reference enters that donor bank.

The backend loads [Meta's SeamlessM4T v2 large checkpoint](https://huggingface.co/facebook/seamless-m4t-v2-large) at revision `5f8cc790b19fc3f67a61c105133b20b34e3dcb76`. The JSON configurations point to its Hugging Face repository ID, so Transformers downloads and caches the pinned files on the first benchmark run. To fetch them before starting a long run, use:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/seamless-m4t-v2-large",
    revision="5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
)
PY
```

The model and corpus have their own usage terms; review their linked source pages before redistribution or commercial use. The benchmark needs a CUDA PyTorch installation and enough disk space for the corpus, Hugging Face model cache, encoded speech cache, and results. Cache preparation and the complete training grid are substantial jobs.

## Run the speech benchmark

The five-method configuration is `live_translation/configs/full_ar.json`. `live_translation/configs/full_ar_rnn.json` also trains and evaluates the causal RNN. The driver runs one configuration and one seed: it prepares the encoded cache, trains the configured defenders, and evaluates every configured defender/attack pair. To run several seeds, invoke it separately with a distinct `--seed` and `--output` for each:

```bash
bash live_translation/scripts/run_benchmark.sh \
  --config live_translation/configs/full_ar.json
```

For the three protocol seeds, run one seed at a time with a separate result directory:

```bash
for seed in 11 22 33; do
  bash live_translation/scripts/run_benchmark.sh \
    --config live_translation/configs/full_ar.json \
    --seed "$seed" --output "live_translation/runs/full_ar/seed_$seed"
done
```

For one seed or a parameter study, call the benchmark directly and give each setting its own output directory:

```bash
python -m live_translation.benchmark \
  --config live_translation/configs/full_ar_rnn.json \
  --stage all --output live_translation/runs/lam2_seed7 \
  --lambda 2 --seed 7 --warmup-steps 100 --defender-steps 100 \
  --eval-fit-steps 100 --attack-inner-steps 3 \
  --picnn-solver-steps 12 --train-duchi-steps 3 \
  --eval-duchi-steps 5 --eval-adaptive-causal-duchi-steps 6
```

`--stage` accepts `prepare`, `train`, `evaluate`, or `all`. The `prepare` stage encodes speech and can be resumed. `train` saves the clean warmup and defender checkpoints. `evaluate` fits fresh learned attackers against each frozen defender, then scores all configured test segments. Run `train` before `evaluate` for a new output directory. A single complete pass over the full training split is the default for warmup, each defender, and each evaluation attacker fit. Explicit `*_steps` values override those epoch counts. Audio longer than 20 seconds uses singleton minibatches; the rest use batches of up to four.

Training's **outer** count is `defender_steps` or `defender_epochs`; `attack_inner_steps` is the learned attack updates within each defender update. For evaluation, `eval_fit_steps` or `eval_fit_epochs` controls how long a learned attacker is fitted against a frozen defender. At inference, `picnn_solver_steps` controls the PICNN's inner solve; `eval_duchi_steps` and `eval_adaptive_causal_duchi_steps` control the two direct attacks. `adaptive_causal_duchi_restarts` adds candidate restarts at each observed prefix. These inference controls do not add defender training updates. After preparing the data, `python -m live_translation.benchmark --config live_translation/configs/full_ar.json --check` validates the input paths and configuration without starting a run.

The examples above change optimizer counts to make the controls visible; those settings are **not** the full-corpus protocol. Keep the original configuration and use multiple seeds for a benchmark comparison. Changing `lam`, a method list, model precision, or a training count creates a new experiment and requires a new output path. The complete full-corpus grid has not been run as part of this folder's preparation, so no full-corpus scores are claimed here.

### Methods and information available to them

| Name | What it does | Main controls |
| --- | --- | --- |
| `nominal` / `clean` | Train the adapter or evaluate it on unchanged speech states. | `warmup_steps`, `defender_steps`, `defender_lr` |
| `causal_picnn` | Learn an attack from the current speech prefix and earlier chosen states. | `attack_lr`, `attack_inner_steps`, `picnn_width`, `picnn_depth`, `picnn_solver_steps`, `picnn_initialization` |
| `anticipative_picnn` | Learn an attack using the complete source utterance, without its reference. | Same PICNN controls |
| `causal_rnn` | Use a GRU to emit each perturbation from observed states and a fixed stage coordinate. | `rnn_hidden_dim`, `rnn_head_dim`, `rnn_time_scale`, `attack_lr`, `attack_inner_steps` |
| `adaptive_causal_duchi` | At each observed prefix, optimize the current action over training-donor continuations, then commit it. | `conditional_samples`, `conditional_shortlist`, `adaptive_causal_duchi_restarts`, `duchi_max_evaluations`, `train_duchi_steps`, `eval_adaptive_causal_duchi_steps` |
| `anticipative_duchi` | Optimize a full path using the complete scored reference. This is a label-informed diagnostic. | `train_duchi_steps`, `eval_duchi_steps`, `duchi_step_size`, `attack_tolerance` |

The learned evaluation attackers are fitted afresh for each frozen defender and `wait_k`, using training examples. Development payoff selects their saved checkpoint; the test set is reserved for final scoring. The finite PICNN, RNN, and Duchi optimizers produce attack candidates, not certified worst cases.

### Important configuration fields

Edit a copy of the JSON file for parameters that do not have a command-line override. The defaults below are from the full Arabic configurations.

| Field | Default | Meaning |
| --- | --- | --- |
| `lam`, `cost_scale` | `3`, `1` | Transport penalty and its fixed denominator. Training and evaluation scaling are `none`. |
| `seed`, `wait_k` | `11`, `[5]` | Random seed and source blocks initially available to the wait-k decoder. |
| `chunk_ms`, `encoder_context_ms` | `640`, `20480` | Speech block size and rolling observed-audio window used to commit a block. |
| `warmup_epochs`, `defender_epochs`, `eval_fit_epochs` | `1`, `1`, `1` | Full shuffled passes when the matching `*_steps` field is unset. |
| `batch_size`, `eval_batch_size`, `training_singleton_seconds` | `4`, `1`, `20` | Training and evaluation batch sizes; longer training rows are processed alone. |
| `defender_lr`, `attack_lr`, `adapter_rank` | `2e-4`, `1e-6`, `4` | Adam learning rates and decoder adapter rank. |
| `picnn_width`, `picnn_depth`, `picnn_solver_steps` | `64`, `3`, `8` | Convex attack network and inner conjugate-solver work. |
| `rnn_hidden_dim`, `rnn_head_dim`, `rnn_time_scale` | `64`, `64`, `128` | Causal RNN architecture and time coordinate. |
| `train_duchi_steps`, `eval_duchi_steps`, `eval_adaptive_causal_duchi_steps` | `2`, `3`, `4` | Direct attack work in training and test evaluation. |
| `conditional_samples`, `conditional_shortlist`, `adaptive_causal_duchi_restarts` | `2`, `128`, `2` | Donors retained, approximate search shortlist, and causal restart count. |
| `eval_selection`, `eval_selection_every_epochs` | `development_payoff`, `1` | Choose a frozen evaluation attacker by development payoff. |
| `max_new_tokens`, `evaluation_max_seconds` | `512`, `100` | Reference-free generation cap and supported evaluation audio horizon. |

`encoder_context_ms` changes the speech encoder's input window. Earlier committed blocks stay fixed when later audio arrives. The donor shortlist is an approximate search over the full training population. Generation stops at EOS or `max_new_tokens`; unfinished outputs are recorded.

## Read the results

Each run saves its resolved `config.json`, `run_identity.json`, training checkpoints, attacker checkpoints, and one JSON condition file per defender, attack, and wait-k value. Condition records retain the utterance ID, official split, hypothesis, reference, cross entropy, transport cost, generation completion, and emission delays. The condition summary reports quality, cross entropy, cost, and latency, including per-split summaries. Penalized payoff is `cross entropy − lam × cost`. Run identity checks guard against accidentally resuming with different code, cache, or model assets.

The package also contains a legacy En–De/fairseq pilot in `backend.py`, `data.py`, and `simuleval_agent.py`. It needs the separately supplied MuST-C En–De release, an official fairseq checkpoint, and a compatible fairseq/SimulEval environment. The CPU audit and En–Ar benchmark above are the supported starting points for this folder.
