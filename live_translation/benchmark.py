"""Run the resumable English–Arabic streaming translation benchmark.

Encoded speech is cached by utterance on the CPU. Each learned evaluation
attack is fitted separately against its defender using training speech.
The direct anticipative baseline can use test references; deployable causal
attacks cannot.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, field, replace
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np
import torch

from .attacks import (AttackConfig, PICNNAttacker, RecurrentAttacker,
                      anticipative_duchi_attack, transport_cost,
                      train_picnn_step, train_recurrent_step)
from .benchmark_analysis import DEFENDERS, ATTACKS, summarize_records
from .conditional import PrefixContinuationBank, streaming_duchi
from .runner import frozen, sha256, write_json

FULL_MANIFEST_ROOT = "live_translation/data/kaggle_mustc_en_ar/full_manifests"
FULL_POPULATION_COUNTS = {"train": 212085, "bank": 212085, "dev": 1073,
                          "test": 2597, "tst-COMMON": 2019, "tst-HE": 578}
LEARNED_ATTACKS = frozenset(('causal_picnn', 'anticipative_picnn', 'causal_rnn'))


@dataclass
class BenchmarkConfig:
    output: str = "live_translation/runs/full_ar/seed_11"
    checkpoint: str = "facebook/seamless-m4t-v2-large"
    revision: str = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
    manifest_root: str = FULL_MANIFEST_ROOT
    cache_root: str = "live_translation/data/kaggle_mustc_en_ar/encoded_640_ctx20480"
    expected_population_counts: dict[str, int] | None = None
    seed: int = 11
    device: str = "cuda"
    chunk_ms: int = 640
    encoder_context_ms: int | None | str = 'auto'
    wait_k: list[int] = field(default_factory=lambda: [5])
    adapter_rank: int = 4
    batch_size: int = 4
    training_singleton_seconds: float = 20.0
    eval_batch_size: int = 1
    warmup_steps: int | None = None
    defender_steps: int | None = None
    warmup_epochs: int = 1
    defender_epochs: int = 1
    defender_lr: float = 2e-4
    attack_lr: float = 1e-6
    attack_inner_steps: int = 1
    eval_fit_steps: int | None = None
    eval_fit_epochs: int = 1
    eval_selection: str = 'development_payoff'
    eval_selection_every: int = 20
    eval_selection_every_epochs: int = 1
    picnn_width: int = 64
    picnn_depth: int = 3
    picnn_solver_steps: int = 8
    rnn_hidden_dim: int = 64
    rnn_head_dim: int = 64
    rnn_time_scale: int = 128
    picnn_architecture: str = 'lastquad'
    picnn_initialization: str = 'conditional_gradient'
    picnn_init_examples: int | None = None
    picnn_init_ridge: float = .01
    picnn_init_damping: float = .25
    decoder_dtype: str = 'float32'
    allow_tf32: bool = False
    train_duchi_steps: int = 2
    eval_duchi_steps: int = 3
    eval_adaptive_causal_duchi_steps: int = 4
    duchi_step_size: float | None = None
    conditional_samples: int = 2
    conditional_shortlist: int = 128
    adaptive_causal_duchi_restarts: int = 2
    evaluation_max_seconds: float = 100.0
    duchi_max_evaluations: int = 10000
    attack_tolerance: float = 1e-4
    lam: float = 3.0
    rho: float | None = None
    cost_scale: float = 1.0
    max_new_tokens: int = 512
    checkpoint_every: int = 500
    training_radius_rule: str = 'none'
    evaluation_scaling: str = 'none'
    defenders: list[str] = field(default_factory=lambda: list(DEFENDERS))
    attacks: list[str] = field(default_factory=lambda: list(ATTACKS))

    def __post_init__(self):
        # Older subset configurations used all available audio. Preserve that
        # behavior while limiting context for the full-corpus default.
        if self.encoder_context_ms == 'auto':
            self.encoder_context_ms = (20480 if Path(self.manifest_root).resolve() ==
                                       Path(FULL_MANIFEST_ROOT).resolve() else None)

    def validate(self):
        if self.seed < 0:
            raise ValueError('seed must be nonnegative')
        for name in ('batch_size','eval_batch_size','adapter_rank',
                     'warmup_epochs','defender_epochs','eval_fit_epochs',
                     'eval_selection_every_epochs','eval_selection_every',
                     'attack_inner_steps','picnn_width','picnn_depth','picnn_solver_steps',
                     'rnn_hidden_dim','rnn_head_dim','rnn_time_scale',
                     'train_duchi_steps','eval_duchi_steps',
                     'eval_adaptive_causal_duchi_steps','adaptive_causal_duchi_restarts',
                     'conditional_samples','conditional_shortlist','checkpoint_every'):
            if getattr(self,name) < 1:
                raise ValueError(f'{name} must be positive')
        for name in ('warmup_steps', 'defender_steps', 'eval_fit_steps'):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f'{name} must be positive when specified')
        if (self.chunk_ms < 1 or (self.encoder_context_ms is not None and
                (not isinstance(self.encoder_context_ms,int) or
                 self.encoder_context_ms < self.chunk_ms or
                 self.encoder_context_ms % self.chunk_ms))):
            raise ValueError('encoder_context_ms must be a positive whole number of audio chunks')
        if self.max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        if not math.isfinite(self.training_singleton_seconds) or self.training_singleton_seconds <= 0:
            raise ValueError('training_singleton_seconds must be positive and finite')
        if not self.wait_k or any(k < 1 for k in self.wait_k) or len(set(self.wait_k)) != len(self.wait_k):
            raise ValueError('wait_k must contain distinct positive chunk counts')
        for name in ('lam','defender_lr','attack_lr','attack_tolerance',
                     'evaluation_max_seconds'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.rho is not None and (not math.isfinite(self.rho) or self.rho <= 0):
            raise ValueError('rho must be positive and finite when specified')
        if self.training_radius_rule != 'none' and (self.rho is None or self.rho<=0):
            raise ValueError('Radius scaling requires a positive radius')
        if not math.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError('Cost scale must be positive and finite')
        if self.duchi_step_size is not None and (
                not math.isfinite(self.duchi_step_size) or self.duchi_step_size <= 0):
            raise ValueError('duchi_step_size must be positive and finite when specified')
        if not self.defenders or not self.attacks:
            raise ValueError('At least one defender and attack are required')
        if set(self.defenders)-set(DEFENDERS) or set(self.attacks)-set(ATTACKS):
            raise ValueError('Unknown experimental method')
        if self.training_radius_rule not in ('offline_global_minibatch_fit', 'none'):
            raise ValueError('Unknown training scaling rule')
        if self.evaluation_scaling not in ('development_radius', 'none'):
            raise ValueError('Unknown evaluation scaling rule')
        if (self.training_radius_rule == 'none') != (self.evaluation_scaling == 'none'):
            raise ValueError('Penalized runs must disable both training and evaluation scaling')
        if self.picnn_architecture not in ('legacy', 'lastquad'):
            raise ValueError('Unknown PICNN architecture')
        if self.picnn_initialization not in ('random', 'conditional_gradient'):
            raise ValueError('Unknown PICNN initialization')
        if (self.picnn_init_examples is not None and self.picnn_init_examples < 1) or self.picnn_init_ridge <= 0 or not 0 < self.picnn_init_damping <= 1:
            raise ValueError('Invalid PICNN initialization settings')
        if self.decoder_dtype not in ('bfloat16', 'float32'):
            raise ValueError('Unsupported decoder precision')
        if self.eval_selection not in ('last', 'development_payoff') or self.eval_selection_every < 1:
            raise ValueError('Invalid evaluation attacker selection')
        if self.duchi_max_evaluations < 1:
            raise ValueError('Duchi evaluation budget must be positive')
        if self.expected_population_counts is not None:
            if (set(self.expected_population_counts) != set(FULL_POPULATION_COUNTS)
                    or any(not isinstance(value,int) or value < 1
                           for value in self.expected_population_counts.values())
                    or self.expected_population_counts['test'] !=
                       self.expected_population_counts['tst-COMMON']+
                       self.expected_population_counts['tst-HE']):
                raise ValueError('Invalid expected full-corpus population counts')


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def fit_offline_training_scale(x, raw, valid, rho, cost_scale=1.0):
    """Fit one shared attack scale from the current training batch.

    Once fitted, the scale stays fixed for that batch's candidate attack.
    Evaluation fits its own scale on development data. The target applies to
    the batch mean, so individual examples can exceed it.
    """
    mean=float(transport_cost(x,raw,valid,cost_scale).mean())
    if not math.isfinite(mean) or mean <= 1e-16:
        raise FloatingPointError('Cannot fit positive training radius to a zero/nonfinite attack')
    return rho/math.sqrt(mean), mean


class TrainingUnion:
    """A stable ID-deduplicated view; all methods see the same training data."""
    def __init__(self, *datasets):
        self.records, self.locations = [], []
        seen = {}
        for dataset in datasets:
            for i, record in enumerate(dataset.records):
                if record['id'] not in seen:
                    self.records.append(record)
                    self.locations.append((dataset, i))
                    seen[record['id']]=record
                elif seen[record['id']] != record:
                    raise ValueError('Conflicting records share a training utterance ID')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        dataset, offset = self.locations[index]
        return dataset[offset]


class Benchmark:
    def __init__(self, config):
        config.validate()
        self.c = config
        self.root = Path(config.output)
        self.root.mkdir(parents=True, exist_ok=True)
        config_path = self.root/'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != asdict(config):
            raise ValueError('Existing run configuration differs; select a new output')
        write_json(config_path, asdict(config))
        seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        torch.set_num_threads(4)
        self.started = time.monotonic()
        self.backend = None
        self._sample_permutations = {}
        self._training_batch_count = None

    def bind_identity(self):
        """Prevent mixed-code or mixed-cache resume of a scientific result."""
        source_dir=Path(__file__).resolve().parent
        source = {f'live_translation/{p.name}':sha256(p) for p in sorted(source_dir.glob('*.py'))}
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            transformers_version = None
        identity = dict(config=asdict(self.c), source_hashes=source,
            cache_fingerprints={name:data.fingerprint for name,data in self.datasets.items()},
            backend=self.backend.provenance,
            runtime_versions={"torch": torch.__version__,
                              "transformers": transformers_version})
        path = self.root/'run_identity.json'
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError('Run source/cache identity changed; do not mix old results with new execution')
        write_json(path,identity)
        for name in source:
            target=self.root/'source'/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copy2(source_dir/Path(name).name,target)
            if sha256(target)!=source[name]:
                raise ValueError('Archived execution source was modified')

    def log(self, stage, **values):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-self.started, **values)
        with (self.root/'trace.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def load_backend(self):
        if self.backend is None:
            from .multilingual_backend import SeamlessM4TBackend
            self.backend = SeamlessM4TBackend(self.c.checkpoint, device=self.c.device,
                chunk_ms=self.c.chunk_ms, target_language='arb', revision=self.c.revision,
                encoder_context_ms=self.c.encoder_context_ms)
            if self.c.decoder_dtype != 'bfloat16':
                self.backend.configure_decoder_precision(self.c.decoder_dtype)
            self.encoding_provenance = dict(self.backend.provenance)
            if not (self.root/'provenance.json').exists():
                write_json(self.root/'provenance.json',dict(**self.backend.provenance,
                    torch_version=torch.__version__,
                    gpu=(torch.cuda.get_device_name(torch.device(self.c.device).index or 0)
                         if torch.device(self.c.device).type == 'cuda' and torch.cuda.is_available()
                         else None),
                    source_hashes={str(p):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
        return self.backend

    def prepare(self):
        from .benchmark_data import cache_manifest
        self.load_backend()
        for split in ('train','bank','dev','test'):
            if split == 'bank' and self.bank_aliases_train():
                self.log('cache_reused', split='bank', source='train')
                continue
            self.log('cache_start', split=split)
            cache_manifest(self.backend, Path(self.c.manifest_root)/f'benchmark_{split}.jsonl',
                           Path(self.c.cache_root)/split)
            self.log('cache_done', split=split)

    def bank_aliases_train(self):
        root = Path(self.c.manifest_root)
        return ((root/'benchmark_train.jsonl').resolve() ==
                (root/'benchmark_bank.jsonl').resolve())

    def load_data(self):
        from .benchmark_data import EncodedDataset, backend_provenance_matches, collate_batches
        self.load_backend()
        self.collate = collate_batches
        self.datasets = {name: EncodedDataset(Path(self.c.cache_root)/(
                            'train' if name == 'bank' and self.bank_aliases_train() else name))
                         for name in ('train','bank','dev','test')}
        for name, data in self.datasets.items():
            if not backend_provenance_matches(data.provenance, self.encoding_provenance):
                raise ValueError(f'{name} encoding provenance differs from active backend')
            if any(r.get('language') != 'en-ar' for r in data.records):
                raise ValueError('The Arabic benchmark requires English-Arabic manifests')
        expected=self.c.expected_population_counts
        if expected is None and Path(self.c.manifest_root).resolve() == Path(FULL_MANIFEST_ROOT).resolve():
            expected=FULL_POPULATION_COUNTS
        if expected is not None:
            from collections import Counter
            actual={name:len(data) for name,data in self.datasets.items()}
            actual.update(Counter(row.get('split',row['id'].split(':',1)[0])
                                  for row in self.datasets['test'].records))
            if actual != expected:
                raise ValueError(f'Full-corpus population differs: expected {expected}, got {actual}')
        self.bind_identity()
        # The donor bank may share training talks but must be separate from dev and test.
        talks = {key:{r['talk_id'] for r in data.records} for key,data in self.datasets.items()}
        if (talks['train']|talks['bank']) & (talks['dev']|talks['test']) or talks['dev']&talks['test']:
            raise ValueError('Official talk-level data separation violated')
        # Keep the donor bank on disk and load only selected trajectories for
        # each observed prefix.
        self.bank = self.datasets['bank']
        # The configured horizon is an upper bound. Donors only need to cover
        # prefixes that the actual development and test rows can expose.
        longest_evaluation_seconds = max(
            float(row['duration'])
            for split in ('dev', 'test') for row in self.datasets[split].records)
        required_chunks=math.ceil(longest_evaluation_seconds*1000/self.c.chunk_ms)
        if hasattr(self.bank, 'lengths'):
            eligible_donors = int((self.bank.lengths >= required_chunks).sum())
            max_chunks = int(self.bank.lengths.max())
        else:
            durations = [float(r['duration']) for r in self.datasets['bank'].records]
            eligible_donors = sum(math.ceil(duration*1000/self.c.chunk_ms) >= required_chunks
                                  for duration in durations)
            max_chunks = math.ceil(max(durations)*1000/self.c.chunk_ms)
        if eligible_donors < self.c.conditional_samples:
            raise ValueError('Training bank must supply enough donors through the longest development/test segment')
        if any(float(r['duration'])>self.c.evaluation_max_seconds+1e-6
               for split in ('dev','test') for r in self.datasets[split].records):
            raise ValueError('Evaluation source exceeds the declared supported horizon')
        if not self.bank_aliases_train():
            self.datasets['train']=TrainingUnion(self.datasets['train'],self.datasets['bank'])
        self.trainable = self.backend.configure_adapters(self.c.adapter_rank)
        self.backend.eval()
        self.initial = self.backend.adapter_state_dict()
        self.log('loaded', sizes={k:len(v) for k,v in self.datasets.items()},
                 trainable_parameters=sum(p.numel() for p in self.trainable),
                 continuation_max_chunks=max_chunks,
                 donor_bank_mode='materialized' if hasattr(self.bank,'states') else 'disk_backed')

    def batch(self, split, indices):
        return self.collate([self.datasets[split][int(i)] for i in indices],
                            self.backend.pad_id).to(self.c.device)

    def training_batches_per_epoch(self):
        data=self.datasets['train']
        if len(data) < 1:
            raise ValueError('Training set is empty')
        signature=(id(data),len(data),self.c.batch_size,self.c.training_singleton_seconds)
        if self._training_batch_count is not None and self._training_batch_count[0]==signature:
            return self._training_batch_count[1]
        if not hasattr(data,'records'):
            batches=math.ceil(len(data)/self.c.batch_size)
        else:
            long_count=sum(float(row['duration']) > self.c.training_singleton_seconds
                           for row in data.records)
            batches=long_count+math.ceil((len(data)-long_count)/self.c.batch_size)
        self._training_batch_count=(signature,batches)
        return batches

    def sample(self, step, *, stream=0, phase=None):
        """One seeded permutation per epoch; every training row appears once.

        Permutations depend only on the stream, epoch and run seed, so a saved
        optimizer step resumes with exactly the same next minibatch. Long
        utterances receive singleton batches to avoid padding a several-minute
        segment alongside three other examples. One plan per active stream is
        retained; memory is linear in the training population.
        """
        count = len(self.datasets['train'])
        if count < 1 or step < 0:
            raise ValueError('Sampling requires a nonempty training set and nonnegative step')
        batches = self.training_batches_per_epoch()
        epoch, within = divmod(step, batches)
        cached = self._sample_permutations.get(stream)
        if cached is None or cached[0] != epoch or cached[2] != count:
            generator = torch.Generator().manual_seed(self.c.seed + 1000000*stream + epoch)
            permutation = torch.randperm(count, generator=generator).tolist()
            rows = getattr(self.datasets['train'],'records',None)
            if rows is None:
                groups=[permutation[i:i+self.c.batch_size]
                        for i in range(0,count,self.c.batch_size)]
            else:
                short=[];long=[]
                for index in permutation:
                    (long if float(rows[index]['duration']) > self.c.training_singleton_seconds
                     else short).append(index)
                groups=[short[i:i+self.c.batch_size]
                        for i in range(0,len(short),self.c.batch_size)]
                groups.extend([[index] for index in long])
                order=torch.randperm(len(groups),generator=generator).tolist()
                groups=[groups[i] for i in order]
            cached = (epoch, groups, count)
            self._sample_permutations[stream] = cached
        indices = cached[1][within]
        return self.batch('train',indices)

    def total_steps(self, phase):
        """Use complete epochs by default; reject a short full-corpus step budget."""
        if phase not in ('warmup', 'defender', 'eval_fit'):
            raise ValueError(f'Unknown training phase: {phase}')
        explicit = getattr(self.c, f'{phase}_steps')
        if explicit is not None:
            if (self.c.expected_population_counts is not None or
                    Path(self.c.manifest_root).resolve() == Path(FULL_MANIFEST_ROOT).resolve()):
                minimum = self.training_batches_per_epoch()
                if explicit < minimum:
                    raise ValueError(f'{phase}_steps={explicit} covers less than one complete '
                                     f'training epoch ({minimum} minibatches); use {phase}_epochs '
                                     'or increase the step count')
            return explicit
        if not hasattr(self, 'datasets'):
            raise RuntimeError('Epoch schedule needs a loaded training set')
        batches = self.training_batches_per_epoch()
        return getattr(self.c, f'{phase}_epochs') * batches

    def selection_interval(self):
        if self.c.eval_fit_steps is not None:
            return self.c.eval_selection_every
        if not hasattr(self, 'datasets'):
            raise RuntimeError('Epoch selection interval needs a loaded training set')
        batches = self.training_batches_per_epoch()
        return self.c.eval_selection_every_epochs * batches

    def iter_batches(self, split):
        for start in range(0,len(self.datasets[split]),self.c.eval_batch_size):
            yield self.batch(split,range(start,min(start+self.c.eval_batch_size,len(self.datasets[split]))))

    def picnn(self, name, *, stream=0):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.c.seed+1000*stream)
            model=PICNNAttacker(self.backend.state_dim, context_dim=self.c.picnn_width,
                hidden_dim=self.c.picnn_width, depth=self.c.picnn_depth,
                threat=name.split('_')[0],solver_steps=self.c.picnn_solver_steps,
                solver_tolerance=self.c.attack_tolerance,architecture=self.c.picnn_architecture)
        return model.to(self.c.device)

    def rnn(self, *, stream=0):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.c.seed+1000*stream)
            model=RecurrentAttacker(self.backend.state_dim,
                hidden_dim=self.c.rnn_hidden_dim,head_dim=self.c.rnn_head_dim,
                time_scale=self.c.rnn_time_scale)
        return model.to(self.c.device)

    def learned_attacker(self, name, *, stream=0):
        if name.endswith('picnn'):
            return self.picnn(name,stream=stream)
        if name=='causal_rnn':
            return self.rnn(stream=stream)
        raise ValueError(f'Not a learned attack: {name}')

    def attack_config(self, evaluation=False, *, causal=False):
        steps = (self.c.eval_adaptive_causal_duchi_steps if causal else self.c.eval_duchi_steps) if evaluation else self.c.train_duchi_steps
        return AttackConfig(lam=self.c.lam, cost_scale=self.c.cost_scale,
            steps=steps,
            step_size=self.c.duchi_step_size,
            tolerance=self.c.attack_tolerance, backtracking_steps=16)

    def raw_attack(self, name, batch, k, attacker=None, evaluation=False):
        if name=='clean': return batch.states,dict(method='identity')
        with frozen(self.backend):
            if name in LEARNED_ATTACKS:
                return attacker(batch.states,batch.mask,self.c.lam,self.c.cost_scale)
            if name=='anticipative_duchi':
                result=anticipative_duchi_attack(batch.states,batch.mask,
                    lambda y:self.backend.loss(y,batch,k),self.attack_config(evaluation))
                result.diagnostics['reference_access']='current_complete_reference'
                return result.y,result.diagnostics
            if name=='adaptive_causal_duchi':
                sampler=PrefixContinuationBank(self.bank,self.backend,k,self.c.conditional_samples,
                                               shortlist=self.c.conditional_shortlist)
                return streaming_duchi(batch.states,batch.mask,sampler,self.attack_config(evaluation,causal=True),
                    solver='adaptive',max_evaluations=self.c.duchi_max_evaluations,
                    restarts=self.c.adaptive_causal_duchi_restarts,seed=self.c.seed)
        raise ValueError(name)

    def attacker_step(self, attacker, optimizer, batch, k):
        with frozen(self.backend):
            train_step = train_recurrent_step if isinstance(attacker, RecurrentAttacker) else train_picnn_step
            return train_step(attacker,optimizer,batch.states,batch.mask,
                lambda y:self.backend.loss(y,batch,k),self.attack_config(),
                gradient_scale=len(batch.states)/self.c.batch_size)

    def initialize_attacker(self, attacker, k):
        if self.c.picnn_initialization == 'random':
            return
        from .initialization import initialize_conditional_shift
        n = (len(self.datasets['train']) if self.c.picnn_init_examples is None
             else min(self.c.picnn_init_examples, len(self.datasets['train'])))
        # Reuse the epoch permutation. The sufficient statistics in the
        # initializer are accumulated online; no corpus tensor is materialized.
        if self.c.picnn_init_examples is not None:
            generator=torch.Generator().manual_seed(self.c.seed+71000)
            indices=torch.randperm(len(self.datasets['train']),generator=generator)[:n]
            batches=(self.batch('train',indices[i:i+self.c.batch_size])
                     for i in range(0,n,self.c.batch_size))
        else:
            def initialization_batches():
                for step in range(self.training_batches_per_epoch()):
                    yield self.sample(step,stream=71,phase='initializer')
            batches=initialization_batches()
        with frozen(self.backend):
            diagnostic = initialize_conditional_shift(attacker,batches,
                lambda x,batch:self.backend.loss(x,batch,k),self.c.lam,self.c.cost_scale,
                self.c.picnn_init_ridge,self.c.picnn_init_damping)
        self.log('attacker_initialization',threat=attacker.threat,k=k,**diagnostic)

    def calibration(self,name,k,attacker,evaluation=True,limit=None):
        if self.c.evaluation_scaling == 'none':
            return 1., dict(calibration_n=0, scaling='none', lambda_penalty=self.c.lam)
        total,count=0.,0
        for batch in self.iter_batches('dev'):
            raw,_=self.raw_attack(name,batch,k,attacker,evaluation)
            costs=transport_cost(batch.states,raw,batch.mask,self.c.cost_scale)
            total+=float(costs.sum());count+=len(costs)
            if limit is not None and count>=limit: break
        mean=total/count
        if not math.isfinite(mean) or mean<=1e-16:
            raise FloatingPointError(f'Cannot calibrate identity/nonfinite {name} attack: {mean}')
        return self.c.rho/math.sqrt(mean), dict(raw_dev_cost=mean,calibration_n=count,
            target_cost=self.c.rho**2,scaling='one_global_dev_scalar_frozen_before_test')

    def save_training(self,path,step,optimizer,attacker=None,attack_optimizer=None,**extra):
        save_torch(path,dict(completed_steps=step,adapter_state=self.backend.adapter_state_dict(),
            optimizer=optimizer.state_dict(),rng=rng_state(),
            attacker=None if attacker is None else attacker.state_dict(),
            attack_optimizer=None if attack_optimizer is None else attack_optimizer.state_dict(),**extra))

    def warmup(self):
        path=self.root/'warmup.pt'
        self.backend.load_adapter_state(self.initial)
        optimizer=torch.optim.Adam(self.trainable,lr=self.c.defender_lr)
        start=0
        if path.exists():
            state=torch.load(path,map_location=self.c.device)
            self.backend.load_adapter_state(state['adapter_state']);optimizer.load_state_dict(state['optimizer'])
            start=state['completed_steps']
            restore_rng(state['rng'])
        total_steps=self.total_steps('warmup')
        self.log('warmup_plan',training_examples=len(self.datasets['train']),
                 batch_size=self.c.batch_size,steps=total_steps,
                 minimum_epochs=self.c.warmup_epochs,start=start)
        for step in range(start,total_steps):
            batch=self.sample(step,stream=10,phase='warmup');k=self.c.wait_k[step%len(self.c.wait_k)]
            optimizer.zero_grad(set_to_none=True)
            losses=self.backend.loss(batch.states,batch,k)
            loss=losses.mean()
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite warmup loss')
            (losses.sum()/self.c.batch_size).backward()
            grad=float(torch.nn.utils.clip_grad_norm_(self.trainable,5.,error_if_nonfinite=True))
            optimizer.step()
            self.log('warmup',step=step,k=k,ce=float(loss.detach()),gradient_norm=grad)
            if (step+1)%self.c.checkpoint_every==0 or step+1==total_steps:
                self.save_training(path,step+1,optimizer)
        return torch.load(path,map_location='cpu')['adapter_state']

    def train_defender(self,name,warm_state):
        path=self.root/f'defender_{name}.pt'
        self.backend.load_adapter_state(warm_state)
        optimizer=torch.optim.Adam(self.trainable,lr=self.c.defender_lr)
        attacker=self.learned_attacker(name) if name in LEARNED_ATTACKS else None
        if name.endswith('picnn') and not path.exists():
            self.initialize_attacker(attacker,self.c.wait_k[0])
        aopt=torch.optim.Adam(attacker.parameters(),lr=self.c.attack_lr) if attacker else None
        start=0
        if path.exists():
            state=torch.load(path,map_location=self.c.device)
            self.backend.load_adapter_state(state['adapter_state']);optimizer.load_state_dict(state['optimizer'])
            start=state['completed_steps']
            restore_rng(state['rng'])
            if attacker: attacker.load_state_dict(state['attacker']);aopt.load_state_dict(state['attack_optimizer'])
        total_steps=self.total_steps('defender')
        self.log('defender_plan',defender=name,training_examples=len(self.datasets['train']),
                 batch_size=self.c.batch_size,steps=total_steps,
                 minimum_epochs=self.c.defender_epochs,start=start)
        for step in range(start,total_steps):
            began=time.monotonic();k=self.c.wait_k[step%len(self.c.wait_k)]
            batch=self.sample(step,stream=20,phase='defender');scale=0.;raw_cost=0.;diagnostic={}
            if name=='nominal': y=batch.states
            else:
                if attacker:
                    for j in range(self.c.attack_inner_steps):
                        diag=self.attacker_step(attacker,aopt,self.sample(
                            step*self.c.attack_inner_steps+j,stream=30,phase='defender'),k)
                        self.log('training_attacker_fit',defender=name,step=step,inner=j,k=k,**diag)
                raw,diagnostic=self.raw_attack(name,batch,k,attacker,evaluation=False)
                # Fit one scale from the training batch and hold it fixed when
                # applying this attack. Live inference uses the saved scale.
                if self.c.training_radius_rule == 'none':
                    scale = 1.
                    raw_cost = float(transport_cost(batch.states,raw,batch.mask,self.c.cost_scale).mean())
                else:
                    scale,raw_cost=fit_offline_training_scale(batch.states,raw,batch.mask,
                        self.c.rho,self.c.cost_scale)
                y=(batch.states+scale*(raw-batch.states)).detach()
            applied_cost=float(transport_cost(batch.states,y,batch.mask,self.c.cost_scale).mean())
            optimizer.zero_grad(set_to_none=True)
            losses=self.backend.loss(y.detach(),batch,k)
            loss=losses.mean()
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite training loss')
            (losses.sum()/self.c.batch_size).backward()
            grad=float(torch.nn.utils.clip_grad_norm_(self.trainable,5.,error_if_nonfinite=True));optimizer.step()
            self.log('defender',defender=name,step=step,k=k,ce=float(loss.detach()),scale=scale,
                transport_cost=applied_cost,raw_training_cost=raw_cost,
                penalized_objective=float(loss.detach())-self.c.lam*applied_cost,
                scaling_fit=self.c.training_radius_rule,seconds=time.monotonic()-began,gradient_norm=grad,
                attack_diagnostic=diagnostic)
            if (step+1)%self.c.checkpoint_every==0 or step+1==total_steps:
                self.save_training(path,step+1,optimizer,attacker,aopt)
        self.log('defender_finished',defender=name,steps=total_steps,
                 full_training_passes=total_steps//self.training_batches_per_epoch())

    def fit_evaluation_attacker(self,defender,name,k):
        path=self.root/'attackers'/f'{defender}__{name}__k{k}.pt'
        defender_hash=sha256(self.root/f'defender_{defender}.pt')
        attacker=self.learned_attacker(name,stream=40+k)
        if name.endswith('picnn') and not path.exists():
            self.initialize_attacker(attacker,k)
        optimizer=torch.optim.Adam(attacker.parameters(),lr=self.c.attack_lr)
        start=0
        best_state=None;best_score=None;selected_step=None
        if path.exists():
            state=torch.load(path,map_location=self.c.device)
            if state.get('defender_sha256') != defender_hash:
                raise ValueError('Evaluation attacker belongs to a different defender checkpoint')
            attacker.load_state_dict(state['attacker']);optimizer.load_state_dict(state['optimizer'])
            start=state['completed_steps']
            restore_rng(state['rng'])
            best_state=state.get('selected_attacker');best_score=state.get('selected_dev_payoff')
            selected_step=state.get('selected_step')

        def select(step):
            nonlocal best_state,best_score,selected_step
            total=cost_total=clean_total=0.;count=0
            with frozen(self.backend):
                for batch in self.iter_batches('dev'):
                    y,_=attacker(batch.states,batch.mask,self.c.lam,self.c.cost_scale)
                    with torch.no_grad():
                        loss=self.backend.loss(y,batch,k)
                        clean=self.backend.loss(batch.states,batch,k)
                    costs=transport_cost(batch.states,y,batch.mask,self.c.cost_scale)
                    total+=float((loss-self.c.lam*costs).sum())
                    clean_total+=float(clean.sum());cost_total+=float(costs.sum());count+=len(costs)
            score=total/count
            if best_score is None or score>best_score:
                best_score=score;selected_step=step
                best_state={key:value.detach().cpu().clone() for key,value in attacker.state_dict().items()}
            self.log('evaluation_attacker_selection',defender=defender,attack=name,k=k,step=step,
                dev_payoff=score,dev_cost=cost_total/count,dev_gain_over_identity=(total-clean_total)/count,
                selected_step=selected_step)

        if self.c.eval_selection=='development_payoff' and best_state is None:
            select(start)
        total_steps=self.total_steps('eval_fit')
        selection_every=self.selection_interval()
        self.log('evaluation_attacker_plan',defender=defender,attack=name,k=k,
                 training_examples=len(self.datasets['train']) if hasattr(self,'datasets') else None,
                 batch_size=self.c.batch_size,steps=total_steps,
                 minimum_epochs=self.c.eval_fit_epochs,start=start,
                 selection_every_steps=selection_every)
        for step in range(start,total_steps):
            diagnostic=self.attacker_step(attacker,optimizer,
                                          self.sample(step,stream=40+k,phase='eval_fit'),k)
            self.log('evaluation_attacker_fit',defender=defender,attack=name,k=k,step=step,**diagnostic)
            if self.c.eval_selection=='development_payoff' and (
                    (step+1)%selection_every==0 or step+1==total_steps):
                select(step+1)
            if (step+1)%self.c.checkpoint_every==0 or step+1==total_steps:
                save_torch(path,dict(attacker=attacker.state_dict(),optimizer=optimizer.state_dict(),
                    completed_steps=step+1,source='training_cache_only',defender=defender,wait_k=k,
                    selected_attacker=best_state,selected_dev_payoff=best_score,selected_step=selected_step,
                    selection=self.c.eval_selection,
                    rng=rng_state(),defender_sha256=defender_hash))
        if best_state is not None:
            attacker.load_state_dict(best_state)
        attacker.selected_step=selected_step if selected_step is not None else total_steps
        return attacker

    def evaluate_condition(self,defender,name,k,attacker):
        path=self.root/'conditions'/f'{defender}__{name}__k{k}.json'
        defender_hash=sha256(self.root/f'defender_{defender}.pt')
        if path.exists():
            completed=json.loads(path.read_text())
            if completed.get('defender_sha256') != defender_hash:
                raise ValueError('Condition belongs to a different defender checkpoint')
            if name=='causal_rnn':
                artifact=path.with_suffix('.pt')
                if not artifact.exists() or sha256(artifact)!=completed.get('live_attack_artifact_sha256'):
                    raise ValueError('Completed causal RNN condition has a missing or altered live artifact')
            return
        attacker_path=self.root/'attackers'/f'{defender}__{name}__k{k}.pt'
        attacker_hash=sha256(attacker_path) if name in LEARNED_ATTACKS else None
        # One immutable file per completed batch makes resume linear in the
        # result size. Rewriting an accumulated 2,597-row checkpoint on every
        # batch was quadratic and especially costly across the full grid.
        journal=path.with_suffix('.partial')
        metadata=journal/'metadata.json'
        records=[];start=0
        if metadata.exists():
            state=json.loads(metadata.read_text())
            if (state.get('defender_sha256') != defender_hash or
                    state.get('attacker_checkpoint_sha256') != attacker_hash or
                    state.get('test_fingerprint') != self.datasets['test'].fingerprint):
                raise ValueError('Partial evaluation belongs to a different defender, attacker or test population')
            scale=state['scale'];cal=state['calibration']
            for chunk in sorted(journal.glob('batch_*.json')):
                payload=json.loads(chunk.read_text())
                if (payload['start'] != start or payload['stop'] != start+len(payload['records']) or
                        chunk.name != f'batch_{start:08d}.json'):
                    raise ValueError(f'Noncontiguous evaluation journal: {chunk}')
                records.extend(payload['records'])
                start=payload['stop']
        else:
            if journal.exists() and any(journal.iterdir()):
                raise ValueError(f'Evaluation journal has no metadata: {journal}')
            if name=='clean': scale=0.;cal=dict(target_cost=0.,calibration_n=0)
            else: scale,cal=self.calibration(name,k,attacker)
            write_json(metadata,dict(scale=scale,calibration=cal,
                defender_sha256=defender_hash,attacker_checkpoint_sha256=attacker_hash,
                test_fingerprint=self.datasets['test'].fingerprint))
        self.log('condition_start',defender=defender,attack=name,k=k,scale=scale,start=start)
        data=self.datasets['test']
        if start > len(data) or [r['id'] for r in records] != [r['id'] for r in data.records[:start]]:
            raise ValueError('Evaluation journal does not match the complete test manifest order')
        for offset in range(start,len(data),self.c.eval_batch_size):
            began=time.monotonic();stop=min(offset+self.c.eval_batch_size,len(data))
            batch=self.batch('test',range(offset,stop))
            raw,diagnostic=self.raw_attack(name,batch,k,attacker,evaluation=True)
            y=(batch.states+scale*(raw-batch.states)).detach()
            with torch.no_grad():
                losses=self.backend.loss(y,batch,k)
                generations=self.backend.decode(y,batch,k,max_new_tokens=self.c.max_new_tokens)
            costs=transport_cost(batch.states,y,batch.mask,self.c.cost_scale)
            batch_records=[]
            for j,(index,generation) in enumerate(zip(range(offset,stop),generations)):
                row=data.records[index]
                batch_records.append(dict(id=batch.ids[j],split=row.get('split',row['id'].split(':',1)[0]),
                    talk_id=row['talk_id'],
                    segment_index=row['segment_index'],expected_talk_segments=row['expected_talk_segments'],
                    offset=row['offset'],reference=batch.references[j],hypothesis=generation.text,
                    ce=float(losses[j]),cost=float(costs[j]),finished=generation.finished,
                    duration_ms=float(batch.durations_ms[j]),word_delays_ms=generation.word_delays_ms,
                    token_ids=generation.token_ids,token_delays_ms=generation.token_delays_ms,
                    eos_delay_ms=generation.eos_delay_ms))
            if len(batch_records) != stop-offset:
                raise ValueError('Decoder returned fewer translations than test examples')
            write_json(journal/f'batch_{offset:08d}.json',
                       dict(start=offset,stop=stop,records=batch_records))
            records.extend(batch_records)
            self.log('condition_batch',defender=defender,attack=name,k=k,completed=stop,total=len(data),
                     seconds=time.monotonic()-began,attack_diagnostic=diagnostic)
        if len(records) != len(data):
            raise ValueError('Incomplete test population at condition finalization')
        summary=summarize_records(records)
        by_split={split:summarize_records([r for r in records if r['split']==split])
                  for split in sorted({r['split'] for r in records})}
        live_artifact_hash=None
        if name=='causal_rnn':
            if attacker is None:
                raise ValueError('A fitted causal RNN is required for its live artifact')
            save_torch(path.with_suffix('.pt'),dict(
                state_dict={key:value.detach().cpu() for key,value in attacker.state_dict().items()},
                scale=scale,config=asdict(self.c),wait_k=k,threat='causal_rnn',
                defender_sha256=defender_hash,
                attacker_checkpoint_sha256=attacker_hash))
            live_artifact_hash=sha256(path.with_suffix('.pt'))
        write_json(path,dict(seed=self.c.seed,wait_k=k,defender=defender,attack=name,
            lambda_penalty=self.c.lam,cost_scale=self.c.cost_scale,
            attacker_checkpoint_sha256=attacker_hash,
            **({'live_attack_artifact_sha256':live_artifact_hash}
               if live_artifact_hash is not None else {}),
            selected_attacker_step=getattr(attacker,'selected_step',None),
            calibration=dict(scale=scale,**cal),summary=summary,summary_by_split=by_split,
            population=dict(expected=len(data),evaluated=len(records),
                            by_split={split:value['n'] for split,value in by_split.items()}),
            records=records,
            defender_sha256=defender_hash))
        shutil.rmtree(journal)
        self.log('condition_finished',defender=defender,attack=name,k=k,summary=summary)

    def run(self, stage):
        if stage in ('prepare','all'): self.prepare()
        if stage=='prepare': return
        self.load_data()
        warm_state=self.warmup() if stage in ('train','all') else None
        for defender in self.c.defenders:
            if stage in ('train','all'): self.train_defender(defender,warm_state)
            if stage in ('evaluate','all'):
                state=torch.load(self.root/f'defender_{defender}.pt',map_location='cpu')
                if state['completed_steps']!=self.total_steps('defender'): raise ValueError('Incomplete defender training')
                self.backend.load_adapter_state(state['adapter_state'])
                for k in self.c.wait_k:
                    for name in self.c.attacks:
                        exists=(self.root/'conditions'/f'{defender}__{name}__k{k}.json').exists()
                        attacker=self.fit_evaluation_attacker(defender,name,k) if name in LEARNED_ATTACKS and not exists else None
                        self.evaluate_condition(defender,name,k,attacker)
        self.log('run_finished',run_stage=stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON configuration; defaults to the full-corpus benchmark')
    parser.add_argument('--stage', choices=('prepare', 'train', 'evaluate', 'all'), default='all',
                        help='Run the requested stage; all also prepares data')
    parser.add_argument('--output', help='Result directory; a resumed run must use the same settings')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--lambda', dest='lam', type=float,
                        help='Penalty weight on transport cost')
    parser.add_argument('--warmup-steps', type=int,
                        help='Number of clean defender updates (overrides warmup_epochs)')
    parser.add_argument('--warmup-epochs', type=int,
                        help='Complete shuffled training passes for clean defender training')
    parser.add_argument('--defender-steps', type=int,
                        help='Number of defender updates (overrides defender_epochs)')
    parser.add_argument('--defender-epochs', type=int,
                        help='Complete shuffled training passes for each defender')
    parser.add_argument('--eval-fit-steps', type=int,
                        help='Attacker fitting updates per evaluation condition (overrides eval_fit_epochs)')
    parser.add_argument('--eval-fit-epochs', type=int,
                        help='Complete shuffled training passes for each learned evaluation attacker')
    parser.add_argument('--attack-inner-steps', type=int,
                        help='Attacker updates inside each defender update')
    parser.add_argument('--picnn-solver-steps', type=int,
                        help='PICNN forward solver steps during training and evaluation')
    parser.add_argument('--train-duchi-steps', type=int,
                        help='Duchi optimization steps during defender training')
    parser.add_argument('--eval-duchi-steps', type=int,
                        help='Duchi optimization steps during evaluation')
    parser.add_argument('--eval-adaptive-causal-duchi-steps', type=int,
                        help='Adaptive causal Duchi optimization steps during evaluation')
    parser.add_argument('--check', action='store_true',
                        help='Validate configuration and required input paths, then exit')
    args = parser.parse_args()
    for phase in ('warmup', 'defender', 'eval_fit'):
        if (getattr(args, f'{phase}_epochs') is not None and
                getattr(args, f'{phase}_steps') is not None):
            parser.error(f'--{phase.replace("_", "-")}-epochs and '
                         f'--{phase.replace("_", "-")}-steps cannot be combined')
    try:
        config = (BenchmarkConfig(**json.loads(Path(args.config).read_text()))
                  if args.config else BenchmarkConfig())
        overrides = {name: getattr(args, name) for name in (
            'output', 'seed', 'lam', 'warmup_steps', 'warmup_epochs',
            'defender_steps', 'defender_epochs', 'eval_fit_steps',
            'eval_fit_epochs', 'attack_inner_steps', 'picnn_solver_steps',
            'train_duchi_steps', 'eval_duchi_steps',
            'eval_adaptive_causal_duchi_steps') if getattr(args, name) is not None}
        for phase in ('warmup', 'defender', 'eval_fit'):
            if getattr(args, f'{phase}_epochs') is not None:
                overrides[f'{phase}_steps'] = None
        config = replace(config, **overrides)
        config.validate()
        if args.check:
            # A checkpoint may be a local directory or a Hub repository ID.
            if (not Path(config.checkpoint).is_dir() and not
                    re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*', config.checkpoint)):
                raise FileNotFoundError(f'Model checkpoint directory or repository ID: {config.checkpoint}')
            if args.stage in ('prepare', 'all'):
                for split in ('train', 'bank', 'dev', 'test'):
                    manifest = Path(config.manifest_root) / f'benchmark_{split}.jsonl'
                    if not manifest.is_file():
                        raise FileNotFoundError(f'Missing {split} manifest: {manifest}')
            else:
                for split in ('train', 'bank', 'dev', 'test'):
                    cached_split = 'train' if split == 'bank' and (
                        Path(config.manifest_root, 'benchmark_bank.jsonl').resolve() ==
                        Path(config.manifest_root, 'benchmark_train.jsonl').resolve()) else split
                    index = Path(config.cache_root, cached_split, 'index.json')
                    if not index.is_file():
                        raise FileNotFoundError(f'Missing {split} encoded cache index: {index}')
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.check:
        print('Configuration and local input paths are valid; remote model IDs are not checked')
        return
    Benchmark(config).run(args.stage)


if __name__ == '__main__':
    main()
