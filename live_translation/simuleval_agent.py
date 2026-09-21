"""Optional current SimulEval speech agent for the committed-prefix backend.

Clean and frozen causal policies run from actual incoming waveform prefixes.
An anticipative attack cannot be evaluated inside this live feed: generate its
offline corrupted representation paths and replay their words/times separately.
The default 5ms audio delivery grid contains the 295ms first chunk boundary.
"""
from __future__ import annotations

import importlib

import torch

from live_translation.backend import FairseqBackend, load_causal_picnn

try:
    from simuleval.agents import SpeechToTextAgent
    from simuleval.agents.actions import ReadAction, WriteAction
    from simuleval.utils import entrypoint
except ImportError as exc:
    raise ImportError("Install SimulEval >=1.1 for this optional agent; its legacy SpeechAgent API is unsupported") from exc


@entrypoint
class CausalRobustSpeechAgent(SpeechToTextAgent):
    """Target subword wait-k, buffering SentencePiece fragments into words."""
    source_segment_size = 5

    @staticmethod
    def add_args(parser):
        parser.add_argument("--live-checkpoint", required=True)
        parser.add_argument("--live-data-bin", required=True)
        parser.add_argument("--live-user-dir", required=True)
        parser.add_argument("--live-config-yaml", default="config_st.yaml")
        parser.add_argument("--live-device", default="cpu")
        parser.add_argument("--live-wait-k", type=int, default=5)
        parser.add_argument("--live-max-new-tokens", type=int, default=128)
        parser.add_argument("--live-adapter-rank", type=int, default=4)
        parser.add_argument("--live-defender-state", help="state_dict saved from this backend (trusted local artifact)")
        parser.add_argument("--live-picnn-state", help="Runner's fitted/calibrated causal PICNN artifact for this wait-k")
        parser.add_argument("--live-rnn-state", help="Runner's fitted/calibrated causal RNN artifact for this wait-k")
        parser.add_argument("--live-acd-state", help="Runner's adaptive_causal_duchi condition JSON; reads its saved training bank and calibrated global scale")
        parser.add_argument("--live-attack-factory", help="Explicit module:function(args,backend) returning a frozen prefix callable; the factory must set information='causal'")

    def __init__(self, args):
        super().__init__(args)
        if args.live_wait_k < 1 or args.live_max_new_tokens < 1:
            raise ValueError("Wait-k and decoding cap must be positive")
        self.backend = FairseqBackend(args.live_checkpoint, args.live_data_bin, args.live_user_dir, args.live_config_yaml, args.live_device)
        if args.live_defender_state:
            self.backend.configure_adapters(args.live_adapter_rank)
            self.backend.load_adapter_checkpoint(args.live_defender_state)
        self.backend.requires_grad_(False).eval()
        self.attack = None
        if sum(bool(getattr(args, name, None)) for name in ("live_picnn_state", "live_rnn_state", "live_acd_state", "live_attack_factory")) > 1:
            raise ValueError("Select one PICNN artifact, RNN artifact, ACD condition artifact, or attack factory")
        if args.live_picnn_state:
            self.attack = load_causal_picnn(args.live_picnn_state, self.backend.state_dim, args.live_device, args.live_wait_k)
        if getattr(args, "live_rnn_state", None):
            from live_translation.rnn_deployment import load_causal_rnn
            self.attack = load_causal_rnn(args.live_rnn_state, self.backend.state_dim,
                                          args.live_device, args.live_wait_k)
        if getattr(args, "live_acd_state", None):
            from live_translation.deployment import load_adaptive_causal_duchi
            self.attack = load_adaptive_causal_duchi(args.live_acd_state, self.backend, args.live_wait_k)
        if args.live_attack_factory:
            module, function = args.live_attack_factory.split(":", 1)
            self.attack = getattr(importlib.import_module(module), function)(args, self.backend)
            if getattr(self.attack, "information", None) != "causal":
                raise ValueError("Live attack factory must explicitly identify a frozen causal policy")
            if isinstance(self.attack, torch.nn.Module):
                self.attack.requires_grad_(False).eval()

    def reset(self):
        super().reset()
        self._received = -1
        self._clean = None
        self._attacked = None
        self._encoder_length = 0
        self._generated = []
        self._pending_word = ""
        self.truncated = False

    def _read_new_chunks(self):
        samples = len(self.states.source)
        if samples == self._received:
            return
        self._received = samples
        rate = self.states.source_sample_rate
        if rate != 16000:
            raise ValueError("Agent requires 16000 Hz source segments")
        if samples < 400:
            if self.states.source_finished:
                raise ValueError("Source shorter than one 25ms analysis window")
            return
        frames = 1 + (samples - 400) // 160
        full_blocks = frames // self.backend.frames_per_chunk
        if not self.states.source_finished and full_blocks == 0:
            return
        previous_count = 0 if self._clean is None else len(self._clean)
        if not self.states.source_finished and full_blocks <= previous_count:
            return
        prefix_samples = samples if self.states.source_finished else (full_blocks * self.backend.frames_per_chunk - 1) * 160 + 400
        waveform = torch.tensor(self.states.source[:prefix_samples], dtype=torch.float32)
        clean, _, real_encoder_length = self.backend.encode_waveform(waveform, rate)
        if previous_count and not torch.allclose(clean[:previous_count], self._clean, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Committed encoder states changed after a future chunk arrived")
        self._clean, self._encoder_length = clean, real_encoder_length
        if self.attack is None:
            attacked = clean
        else:
            valid = torch.ones((1, len(clean)), dtype=torch.bool, device=clean.device)
            # Convex-energy policies may require autograd even when frozen.
            with torch.enable_grad():
                attacked = self.attack(clean[None], valid)
                if hasattr(attacked, "y"):
                    attacked = attacked.y
                elif isinstance(attacked, tuple):
                    attacked = attacked[0]
                attacked = attacked[0].detach()
            if attacked.shape != clean.shape or not bool(torch.isfinite(attacked).all()):
                raise ValueError("Causal policy returned invalid corrupted states")
        if self._attacked is not None and not torch.allclose(attacked[:previous_count], self._attacked, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Attack revised an earlier committed action after seeing future audio")
        self._attacked = attacked

    def policy(self):
        self._read_new_chunks()
        if self._clean is None:
            return ReadAction()
        while True:
            if len(self._generated) >= self.args.live_max_new_tokens:
                self.truncated = True
                text, self._pending_word = self._pending_word, ""
                return WriteAction(text, finished=True)
            requested_chunks = self.args.live_wait_k + len(self._generated)
            if requested_chunks > len(self._clean) and not self.states.source_finished:
                return ReadAction()
            read = min(len(self._clean), requested_chunks)
            encoder_length = min(self._encoder_length, read * self.backend.pre_decision_ratio)
            with torch.no_grad():
                token = self.backend.next_token(self._attacked[:read], self._generated, self.args.live_wait_k, encoder_length)
            if token == self.backend.eos_id:
                text, self._pending_word = self._pending_word, ""
                return WriteAction(text, finished=True)
            self._generated.append(token)
            piece = self.backend.dictionary[token]
            if piece.startswith("▁") and self._pending_word:
                completed = self._pending_word
                self._pending_word = piece.removeprefix("▁")
                return WriteAction(completed, finished=False)
            self._pending_word += piece.removeprefix("▁")
