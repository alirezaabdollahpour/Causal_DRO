"""Explicit wait-k defenders with a shared immutable causal speech encoder.

The fairseq backend initializes from the official MuST-C checkpoint but defines
its own committed-prefix encoder and deterministic subword wait-k attention.
It is consequently not a reproduction of the upstream checkpoint's BLEU/AL.
Optional speech dependencies are imported only when that backend is selected.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import EncodedBatch, ToyVocabulary, make_toy_batch, read_audio, read_manifest


def waitk_mask(target_steps: int, source_steps: int, wait_k: int, device=None) -> Tensor:
    """True exactly for forbidden keys; target indices start at zero."""
    if wait_k < 1:
        raise ValueError("wait_k must be >= 1, measured in source chunks per target subword")
    return torch.arange(source_steps, device=device)[None] >= (wait_k + torch.arange(target_steps, device=device))[:, None]


class LoRALinear(nn.Module):
    """Frozen projection plus alpha/r times a trainable rank-r residual."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("Adapter rank must be positive")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight + self.scale * self.B @ self.A

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)


@dataclass
class Generation:
    text: str
    token_ids: list[int]             # excludes EOS and PAD
    token_delays_ms: list[float]
    word_delays_ms: list[float]      # word completion/emission times, excludes EOS
    finished: bool                  # true iff decoder emitted EOS
    eos_delay_ms: float | None = None


class TranslationBackend(nn.Module):
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def train(self, mode: bool = True):
        # Frozen pretrained dropout must not randomize the reference law/attacks.
        # Adapter parameters receive gradients even while modules are in eval mode.
        super().train(False)
        return self

    def load_adapter_checkpoint(self, path: str | Path):
        """Load the runner's adapter-only artifact after configure_adapters()."""
        payload = torch.load(path, map_location=next(self.parameters()).device, weights_only=True)
        supplied = payload.get("adapter_state")
        expected = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if not expected or not isinstance(supplied, dict) or supplied.keys() != expected.keys():
            raise ValueError("Adapter artifact names do not match configured trainable parameters")
        with torch.no_grad():
            for name, value in supplied.items():
                if expected[name].shape != value.shape:
                    raise ValueError(f"Adapter shape mismatch: {name}")
                expected[name].copy_(value)
        return payload

    def logits(self, states: Tensor, lengths: Tensor, previous_tokens: Tensor, wait_k: int, encoder_lengths: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def loss(self, states: Tensor, batch: EncodedBatch, wait_k: int) -> Tensor:
        targets = batch.target_tokens
        previous = torch.cat([targets.new_full((targets.shape[0], 1), self.bos_id), targets[:, :-1]], dim=1)
        logits = self.logits(states, batch.lengths, previous, wait_k, batch.encoder_lengths)
        token_loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none", ignore_index=self.pad_id)
        valid = torch.arange(targets.shape[1], device=targets.device)[None] < batch.target_lengths[:, None]
        return (token_loss * valid).sum(1) / batch.target_lengths

    def next_token(self, states: Tensor, token_ids: list[int], wait_k: int, encoder_length: int | None = None) -> int:
        """Only an observed state prefix is accepted by streaming callers."""
        if states.ndim != 2 or not len(states):
            raise ValueError("next_token needs a nonempty [observed chunks, dimension] prefix")
        prev = torch.tensor([[self.bos_id, *token_ids]], device=states.device)
        lengths = torch.tensor([len(states)], device=states.device)
        enc_len = None if encoder_length is None else lengths.new_tensor([encoder_length])
        scores = self.logits(states[None], lengths, prev, wait_k, enc_len)[0, -1].clone()
        scores[self.pad_id] = -torch.inf
        if self.bos_id != self.eos_id:
            scores[self.bos_id] = -torch.inf
        return int(scores.argmax())

    def detokenize(self, ids: list[int], delays: list[float], terminal_delay: float) -> tuple[str, list[float]]:
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, states: Tensor, batch: EncodedBatch, wait_k: int, max_new_tokens: int = 128) -> list[Generation]:
        if wait_k < 1 or max_new_tokens <= 0:
            raise ValueError("wait_k and max_new_tokens must be positive")
        generations = []
        for i, length in enumerate(batch.lengths.tolist()):
            tokens, delays, finished, eos_delay = [], [], False, None
            for m in range(max_new_tokens):
                read = min(length, wait_k + m)
                enc_len = None
                if batch.encoder_lengths is not None:
                    enc_len = min(int(batch.encoder_lengths[i]), read * self.pre_decision_ratio)
                token = self.next_token(states[i, :read], tokens, wait_k, enc_len)
                delay = (float(batch.durations_ms[i]) if wait_k + m > length
                         else float(batch.chunk_end_ms[i, read - 1]))
                if token == self.eos_id:
                    finished, eos_delay = True, delay
                    break
                tokens.append(token)
                delays.append(delay)
            terminal = eos_delay if eos_delay is not None else (delays[-1] if delays else float(batch.chunk_end_ms[i, min(length, wait_k) - 1]))
            text, word_delays = self.detokenize(tokens, delays, terminal)
            generations.append(Generation(text, tokens, delays, word_delays, finished, eos_delay))
        return generations


class ToyBackend(TranslationBackend):
    """Small synthetic-vector decoder solely for execution and causality audits."""
    def __init__(self, state_dim: int = 8, vocab_size: int = 11, hidden_dim: int = 16, seed: int = 0):
        super().__init__()
        self.state_dim, self.vocab_size = state_dim, vocab_size
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.pad_id)
            self.history = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.source = nn.Linear(state_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, vocab_size)
        self.words = ToyVocabulary.words(vocab_size)
        self.eval()

    def make_batch(self, count: int, seed: int, prefix: str = "synthetic", min_chunks: int = 4, max_chunks: int = 8) -> EncodedBatch:
        batch = make_toy_batch(count, self.state_dim, seed, min_chunks, max_chunks, self.vocab_size)
        batch.ids = [f"{prefix}:{seed}:{i}" for i in range(count)]
        return batch.to(next(self.parameters()).device)

    def configure_adapters(self, rank: int = 4) -> list[nn.Parameter]:
        self.requires_grad_(False)
        if not isinstance(self.output, LoRALinear):
            self.output = LoRALinear(self.output, rank)
        else:
            self.output.A.requires_grad_(True)
            self.output.B.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    def logits(self, states, lengths, previous_tokens, wait_k, encoder_lengths=None):
        if wait_k < 1:
            raise ValueError("wait_k must be positive")
        b, t, d = states.shape
        u = previous_tokens.shape[1]
        steps = torch.arange(u, device=states.device)
        visible = torch.minimum(lengths[:, None], wait_k + steps[None])
        running = states.cumsum(dim=1)
        context = running.gather(1, (visible - 1)[:, :, None].expand(b, u, d)) / visible[:, :, None]
        # A positional source focus plus prefix summary; every term is available.
        focus_index = torch.minimum(steps[None].expand(b, -1), visible - 1)
        focus = states.gather(1, focus_index[:, :, None].expand(b, u, d))
        history, _ = self.history(self.embedding(previous_tokens))
        return self.output(torch.tanh(self.source(focus + .2 * context) + .1 * history))

    def detokenize(self, ids, delays, terminal_delay):
        return " ".join(self.words[i] for i in ids), list(delays)


class ExplicitWaitKAttention(nn.Module):
    """Deterministic prefix soft attention using learned checkpoint projections.

    We bypass legacy monotonic kernels and impose the source mask explicitly.
    Decoder self-attention remains causal. Source states are immutable, so the
    full teacher-forced pass equals recomputing the observed prefix at each step.
    """
    def __init__(self, original: nn.Module, ratio: int):
        super().__init__()
        self.q_proj = original.q_in_proj["soft"]
        self.k_proj = original.k_in_proj["soft"]
        self.v_proj = original.v_proj
        self.out_proj = original.out_proj
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.scaling = original.scaling
        self.pre_decision_ratio = ratio
        self.wait_k = int(original.waitk_lagging)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, **kwargs):
        if incremental_state is not None:
            raise ValueError("This reference backend uses full-prefix decoding, not legacy incremental caches")
        u, b, dim = query.shape
        s = key.shape[0]
        def heads(x):
            return x.reshape(x.shape[0], b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = heads(self.q_proj(query)), heads(self.k_proj(key)), heads(self.v_proj(value))
        energy = (q * self.scaling) @ k.transpose(-1, -2)
        boundary = (self.wait_k + torch.arange(u, device=query.device)) * self.pre_decision_ratio
        allowed = torch.arange(s, device=query.device)[None] < boundary[:, None]
        mask = ~allowed[None, None]
        if key_padding_mask is not None:
            mask = mask | key_padding_mask[:, None, None, :]
        beta = torch.softmax(energy.masked_fill(mask, -torch.inf), dim=-1)
        output = (beta @ v).permute(2, 0, 1, 3).reshape(u, b, dim)
        return self.out_proj(output), {"p_choose": None, "alpha": None, "beta": beta, "soft_energy": energy}


def commit_prefix_encoder(encoder: nn.Module, features: Tensor, frames_per_chunk: int, states_per_chunk: int) -> tuple[Tensor, int]:
    """Encode each feature prefix and commit its newly exposed block exactly once.

    This works even for a bidirectional encoder: future input never exists in a
    call that creates an earlier committed block. Group padding is decoder-masked.
    """
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Need nonempty [feature frames, channels]")
    blocks, committed, dim = [], 0, None
    for end in range(frames_per_chunk, len(features) + frames_per_chunk, frames_per_chunk):
        end = min(end, len(features))
        prefix = features[:end][None]
        output = encoder(src_tokens=prefix, src_lengths=torch.tensor([end], device=features.device))
        encoded = output["encoder_out"][0][:, 0]
        dim = encoded.shape[-1]
        stop = min(encoded.shape[0], committed + states_per_chunk)
        block = encoded[committed:stop]
        if not len(block):
            raise RuntimeError("Encoder produced no new states at a chunk boundary; unsupported subsampling")
        if end < len(features) and len(block) != states_per_chunk:
            raise RuntimeError("Checkpoint subsampling does not match configured fixed chunks")
        blocks.append(F.pad(block, (0, 0, 0, states_per_chunk - len(block))).reshape(-1))
        committed = stop
    return torch.stack(blocks), committed


class FairseqBackend(TranslationBackend):
    """MuST-C checkpoint adapter with fixed 16 kHz / 80-bin / 4x frontend."""
    def __init__(self, checkpoint: str | Path, data_bin: str | Path, user_dir: str | Path,
                 config_yaml: str = "config_st.yaml", device: str = "cpu"):
        super().__init__()
        for path in (checkpoint, data_bin, user_dir):
            if not Path(path).exists():
                raise FileNotFoundError(path)
        user_path = Path(user_dir).resolve()
        if user_path.name != "simultaneous_translation" or user_path.parent.name != "examples":
            raise ValueError("user_dir must be FAIRSEQ_ROOT/examples/simultaneous_translation")
        checkout_root = str(user_path.parent.parent)
        if checkout_root not in sys.path:
            # Do this before importing fairseq: its eager task imports can
            # otherwise bind an unrelated namespace package named examples.
            sys.path.insert(0, checkout_root)
        try:
            import numpy as np
            import yaml
            import sentencepiece as spm
            from fairseq import checkpoint_utils
        except ImportError as exc:
            raise RuntimeError("FairseqBackend needs a compatible fairseq source checkout, sentencepiece, torchaudio, PyYAML and soundfile; see live_translation/README.md") from exc
        # Upstream models/__init__.py imports "examples.simultaneous_translation"
        # explicitly. Generic import_user_module also loads the same files as
        # "simultaneous_translation", registering architectures twice on this pin.
        # Import exactly their canonical namespace once, from the supplied tree.
        imported = importlib.import_module("examples.simultaneous_translation")
        if Path(imported.__file__).resolve().parent != user_path:
            raise ImportError("A different fairseq examples package was already imported")
        overrides = {"data": str(Path(data_bin).resolve()), "config_yaml": config_yaml,
                     "load_pretrained_encoder_from": None, "load_pretrained_decoder_from": None}
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)], arg_overrides=overrides)
        if len(models) != 1:
            raise ValueError("Expected exactly one pretrained model")
        self.model, self.task = models[0], task
        dictionary = task.target_dictionary
        self.dictionary = dictionary
        self.pad_id, self.bos_id, self.eos_id = dictionary.pad(), dictionary.eos(), dictionary.eos()
        layers = self.model.decoder.layers
        if not layers or not all(hasattr(layer.encoder_attn, "waitk_lagging") for layer in layers):
            raise ValueError("Require a fixed-predecision wait-k checkpoint, not MMA or an offline model")
        self.pre_decision_ratio = int(getattr(layers[0].encoder_attn, "pre_decision_ratio", 1))
        if self.model.encoder.pooling_ratio() != 4:
            raise ValueError("Only the official 4x ConvTransformer frontend is supported")
        for layer in layers:
            if getattr(layer, "cross_self_attention", False):
                raise ValueError("Source-aware decoder self-attention would violate this backend's explicit mask")
            if int(getattr(layer.encoder_attn, "pre_decision_ratio", 1)) != self.pre_decision_ratio:
                raise ValueError("All decoder layers must use the same fixed predecision ratio")
            layer.encoder_attn = ExplicitWaitKAttention(layer.encoder_attn, self.pre_decision_ratio)
        self.encoder_dim = int(self.model.encoder.out.out_features)
        self.state_dim = self.encoder_dim * self.pre_decision_ratio
        self.vocab_size = len(dictionary)
        self.frames_per_chunk = 4 * self.pre_decision_ratio
        self.chunk_ms = self.frames_per_chunk * 10
        config = yaml.safe_load((Path(data_bin) / config_yaml).read_text())
        def resolve(path):
            path = Path(path)
            return path if path.is_absolute() else Path(data_bin) / path
        self.spm = spm.SentencePieceProcessor(model_file=str(resolve(config["bpe_tokenizer"]["sentencepiece_model"])))
        cmvn = np.load(resolve(config["global_cmvn"]["stats_npz_path"]))
        self.register_buffer("cmvn_mean", torch.from_numpy(cmvn["mean"]).float())
        self.register_buffer("cmvn_std", torch.from_numpy(cmvn["std"]).float())
        if not bool((self.cmvn_std > 0).all()):
            raise ValueError("Global CMVN standard deviations must be positive")
        self.provenance = {"backend": "fairseq_committed_prefix_explicit_waitk", "checkpoint": str(Path(checkpoint).resolve()),
                           "checkpoint_sha256": file_sha256(checkpoint), "config_yaml": str(Path(data_bin) / config_yaml),
                           "state_dim": self.state_dim, "pre_decision_ratio": self.pre_decision_ratio,
                           "chunk_ms": self.chunk_ms, "first_complete_chunk_ms": self.chunk_ms + 15,
                           "encoder": "bidirectional prefix recomputation; immutable newly exposed blocks",
                           "schedule_unit": "target SentencePiece subword per source chunk",
                           "upstream_reproduction": False}
        self.to(device).eval()
        self.requires_grad_(False)

    def configure_adapters(self, rank: int = 4) -> list[nn.Parameter]:
        self.requires_grad_(False)
        output = self.model.decoder.output_projection
        if not isinstance(output, LoRALinear):
            self.model.decoder.output_projection = LoRALinear(output, rank)
        else:
            output.A.requires_grad_(True)
            output.B.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]

    def logits(self, states, lengths, previous_tokens, wait_k, encoder_lengths=None):
        if wait_k < 1:
            raise ValueError("wait_k must be positive")
        b, t, _ = states.shape
        flat = states.reshape(b, t * self.pre_decision_ratio, self.encoder_dim)
        real_lengths = lengths * self.pre_decision_ratio if encoder_lengths is None else encoder_lengths
        padding = torch.arange(flat.shape[1], device=flat.device)[None] >= real_lengths[:, None]
        encoder = {"encoder_out": [flat.transpose(0, 1)], "encoder_padding_mask": [padding],
                   "encoder_embedding": [], "encoder_states": [], "src_tokens": [], "src_lengths": []}
        for layer in self.model.decoder.layers:
            layer.encoder_attn.wait_k = wait_k
        return self.model.decoder(prev_output_tokens=previous_tokens, encoder_out=encoder, incremental_state=None)[0]

    def features(self, waveform: Tensor, sample_rate: int = 16000) -> Tensor:
        try:
            from torchaudio.compliance.kaldi import fbank
        except (ImportError, OSError) as exc:
            raise RuntimeError("Install torchaudio built for your torch/CUDA version to encode real speech") from exc
        if sample_rate != 16000 or waveform.ndim != 1 or waveform.numel() < 400:
            raise ValueError("Expected >=25ms mono 16000 Hz waveform")
        # Disable full-utterance DC removal; fbank performs frame-local centering.
        # snip_edges=True prevents future window padding. No utterance CMVN.
        # The pretrained global statistics use fairseq's 16-bit PCM amplitude
        # convention, while SoundFile and SimulEval supply normalized floats.
        features = fbank(waveform.cpu()[None] * (2 ** 15), sample_frequency=sample_rate, num_mel_bins=80,
                         frame_length=25, frame_shift=10, dither=0.0, snip_edges=True)
        features = features.to(self.cmvn_mean.device)
        return (features - self.cmvn_mean) / self.cmvn_std

    @torch.no_grad()
    def encode_waveform(self, waveform: Tensor, sample_rate: int = 16000) -> tuple[Tensor, Tensor, int]:
        features = self.features(waveform, sample_rate)
        states, encoder_length = commit_prefix_encoder(self.model.encoder, features, self.frames_per_chunk, self.pre_decision_ratio)
        ends = torch.arange(1, len(states) + 1, device=states.device).float() * self.chunk_ms + 15
        duration_ms = waveform.numel() * 1000 / sample_rate
        # Complete blocks are available after their final analysis window. Only
        # a partial final block requires source EOS; count that observed time.
        ends[-1] = min(float(ends[-1]), duration_ms)
        return states, ends, encoder_length

    @torch.no_grad()
    def encode(self, manifest: str | Path | list[dict[str, Any]], limit: int | None = None) -> EncodedBatch:
        records = read_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            records = records[:limit]
        if not records:
            raise ValueError("No records to encode")
        xs, times, targets, durations, enc_lengths = [], [], [], [], []
        for record in records:
            waveform, rate = read_audio(record)
            states, ends, enc_length = self.encode_waveform(waveform, rate)
            pieces = " ".join(self.spm.encode(record["target_text"], out_type=str))
            target = self.dictionary.encode_line(pieces, add_if_not_exist=False, append_eos=True).long().to(states.device)
            xs.append(states); times.append(ends); targets.append(target)
            durations.append(waveform.numel() * 1000 / rate); enc_lengths.append(enc_length)
        device = xs[0].device
        lengths = torch.tensor([len(x) for x in xs], device=device)
        target_lengths = torch.tensor([len(y) for y in targets], device=device)
        padded = nn.utils.rnn.pad_sequence(xs, batch_first=True)
        padded_times = nn.utils.rnn.pad_sequence(times, batch_first=True)
        for i, length in enumerate(lengths.tolist()):
            padded_times[i, length:] = times[i][-1]
        return EncodedBatch(padded, lengths, nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_id),
                            target_lengths, padded_times, torch.tensor(durations, device=device),
                            [r["target_text"] for r in records], [r["id"] for r in records],
                            torch.tensor(enc_lengths, device=device), {**self.provenance,
                                "corpus": ",".join(sorted({r.get("corpus", "supplied_audio") for r in records})),
                                "splits": sorted({r.get("split", "unspecified") for r in records})})

    def detokenize(self, ids, delays, terminal_delay):
        pieces = [self.dictionary[i] for i in ids]
        # Buffer a word until the next leading ▁ announces its completion.
        # The next boundary token's read time (not the previous piece's time)
        # is its actual streaming emission time.
        words, word_delays, pending = [], [], ""
        for piece, delay in zip(pieces, delays):
            if piece.startswith("▁") and pending:
                words.append(pending)
                word_delays.append(delay)
                pending = ""
            pending += piece.replace("▁", " ").lstrip() if not pending else piece.replace("▁", "")
        if pending:
            words.append(pending)
            word_delays.append(terminal_delay)
        return " ".join(words), word_delays


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CalibratedCausalPICNN(nn.Module):
    """Frozen fitted PICNN plus one dev-calibrated scalar, valid on prefixes."""
    information = "causal"

    def __init__(self, attacker, scale: float, lam: float, cost_scale: float):
        super().__init__()
        self.attacker = attacker.requires_grad_(False).eval()
        self.scale, self.lam, self.cost_scale = scale, lam, cost_scale

    def forward(self, x, valid):
        y, _ = self.attacker(x, valid, self.lam, self.cost_scale)
        return x + self.scale * (y - x)


def load_causal_picnn(path: str | Path, state_dim: int, device: str, wait_k: int) -> CalibratedCausalPICNN:
    from .attacks import PICNNAttacker

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("threat") != "causal" or payload.get("wait_k") != wait_k:
        raise ValueError("Live PICNN artifact must be causal and fitted at this wait-k")
    config = payload["config"]
    attacker = PICNNAttacker(state_dim, context_dim=config["picnn_width"], hidden_dim=config["picnn_width"],
        depth=config["picnn_depth"], threat="causal", solver_steps=config["picnn_solver_steps"], solver_tolerance=config["attack_tolerance"]).to(device)
    attacker.load_state_dict(payload["state_dict"], strict=True)
    scale = float(payload["scale"])
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Calibrated attack scale must be finite and nonnegative")
    return CalibratedCausalPICNN(attacker, scale, config["lam"], config["cost_scale"])
