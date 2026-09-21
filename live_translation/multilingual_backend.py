"""Arabic speech translation with immutable causal encodings and wait-k attention.

This is an explicit streaming adaptation of SeamlessM4T v2, not the separately
trained SeamlessStreaming checkpoint. The frozen speech encoder commits new
states from observed audio only, using either the complete prefix or a declared
rolling context window. Both adversarial information classes use this same
configured defender.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backend import Generation, LoRALinear, TranslationBackend
from .data import EncodedBatch, read_audio, read_manifest


class FloatAdapterLinear(LoRALinear):
    """Float32 trainable adapters with frozen reduced-precision projections."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__(base, rank)
        self.A.data = self.A.data.float()
        self.B.data = self.B.data.float()

    def forward(self, x):
        base = self.base(x)
        residual = F.linear(F.linear(x.float(), self.A), self.B)
        return base + (self.scale * residual).to(base.dtype)


class SeamlessWaitKAttention(nn.Module):
    """Add a source-prefix mask to the pretrained decoder cross attention."""

    def __init__(self, original: nn.Module, states_per_chunk: int):
        super().__init__()
        self.original = original
        self.states_per_chunk = states_per_chunk
        self.wait_k = 1
        self.position_offset = 0

    def forward(self, hidden_states, encoder_hidden_states=None, past_key_value=None,
                past_key_values=None, attention_mask=None, output_attentions=False,
                **kwargs):
        if encoder_hidden_states is None:
            raise ValueError("Wait-k wrapper must only replace decoder cross attention")
        if past_key_value is not None and past_key_values is not None:
            raise ValueError("Pass only one decoder cache representation")
        # Decoder prefix is [EOS, target-language]. Both compulsory prefix
        # positions see k chunks; position 1 predicts the first lexical token.
        position = torch.arange(hidden_states.shape[1], device=hidden_states.device) + self.position_offset
        writes = (position - 1).clamp_min(0)
        boundary = (self.wait_k + writes) * self.states_per_chunk
        forbidden = torch.arange(encoder_hidden_states.shape[1], device=hidden_states.device)[None] >= boundary[:, None]
        causal_mask = torch.zeros_like(forbidden, dtype=hidden_states.dtype).masked_fill(forbidden, -torch.inf)[None, None]
        mask = causal_mask if attention_mask is None else attention_mask + causal_mask
        cache_argument = ({"past_key_values": past_key_values} if past_key_values is not None
                          else {"past_key_value": past_key_value} if past_key_value is not None
                          else {})
        return self.original(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                             attention_mask=mask, output_attentions=output_attentions,
                             **cache_argument, **kwargs)


class SeamlessM4TBackend(TranslationBackend):
    """Frozen English-speech/Arabic-text checkpoint with decoder LoRA adapters.

    ``states`` stays float32 for transport optimization. Pretrained computation
    uses the requested model dtype. The default 320ms chunk holds two 1024-D
    speech states; its transport vector has dimension 2048. A bounded encoder
    context uses a distinct representation and cache provenance. No references
    enter source encoding, generation stopping, masks, or decoding constraints.
    """

    def __init__(self, checkpoint: str | Path = "facebook/seamless-m4t-v2-large",
                 device: str = "cuda", chunk_ms: int = 320,
                 target_language: str = "arb", dtype: str = "bfloat16",
                 revision: str | None = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
                 encoder_context_ms: int | None = None):
        super().__init__()
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        if chunk_ms < 160 or chunk_ms % 160:
            raise ValueError("SeamlessM4T chunks must be positive multiples of its 160ms output stride")
        if dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported pretrained dtype")
        if encoder_context_ms is not None and (
                isinstance(encoder_context_ms, bool) or not isinstance(encoder_context_ms, int)
                or encoder_context_ms < chunk_ms or encoder_context_ms % chunk_ms):
            raise ValueError("Encoder context must be a positive whole number of source chunks")
        local = Path(checkpoint).is_dir()
        load_args = {} if local else {"revision": revision}
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), **load_args)
        self.tokenizer = self.processor.tokenizer
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            str(checkpoint), torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True, **load_args)
        self.model.to(device).eval().requires_grad_(False)
        cfg = self.model.config
        if cfg.adaptor_stride != 8 or cfg.num_adapter_layers != 1 or self.processor.feature_extractor.stride != 2:
            raise ValueError("Backend requires the v2 2x feature stacking and 8x adapter")
        lang_ids = self.model.generation_config.text_decoder_lang_to_code_id
        if target_language not in lang_ids:
            raise ValueError(f"Unsupported target text language: {target_language}")
        self.language_id = int(lang_ids[target_language])
        self.target_language = target_language
        self.pad_id, self.bos_id, self.eos_id = int(cfg.pad_token_id), int(cfg.decoder_start_token_id), int(cfg.eos_token_id)
        self.encoder_dim = int(cfg.hidden_size)
        self.pre_decision_ratio = chunk_ms // 160
        self.chunk_ms = int(chunk_ms)
        self.encoder_context_ms = encoder_context_ms
        self.state_dim = self.pre_decision_ratio * self.encoder_dim
        self.vocab_size = int(cfg.vocab_size)
        for layer in self.model.text_decoder.layers:
            layer.cross_attention = SeamlessWaitKAttention(layer.cross_attention, self.pre_decision_ratio)
        self.provenance = {
            "backend": "seamless_m4t_v2_committed_prefix_explicit_waitk",
            "checkpoint": str(checkpoint), "checkpoint_revision": revision,
            "target_language": target_language, "target_language_id": self.language_id,
            "pretrained_dtype": dtype, "state_dtype": "float32",
            "state_dim": self.state_dim, "encoder_dim": self.encoder_dim,
            "pre_decision_ratio": self.pre_decision_ratio, "chunk_ms": self.chunk_ms,
            "encoder": "frozen bidirectional observed-prefix recomputation; immutable newly exposed blocks",
            "normalization": "checkpoint mel-bin CMVN recomputed on each observed waveform prefix",
            "short_source_eos_padding": "observed EOS shorter than35ms: synthetic zero pad to35ms, retain real availability",
            "schedule_unit": "one target SentencePiece subword per source chunk",
            "decoder_prompt": "[EOS, __arb__] for Modern Standard Arabic",
            "upstream_reproduction": False,
        }
        if encoder_context_ms is not None:
            # The bounded policy is a distinct representation. In particular,
            # it must not silently reuse full-prefix caches or historical scores.
            self.provenance.update(
                encoder="frozen bidirectional observed-window recomputation; immutable newly exposed blocks",
                normalization="checkpoint mel-bin CMVN recomputed on each observed waveform window",
                encoder_context_ms=encoder_context_ms,
                encoder_window_policy="last complete source chunks plus current observed chunk; left edge aligned to source chunks",
            )
        self.eval()

    @property
    def device(self):
        return self.model.shared.weight.device

    @property
    def model_dtype(self):
        return self.model.shared.weight.dtype

    def configure_decoder_precision(self, dtype: str) -> None:
        """Change translation arithmetic without changing speech cache geometry."""
        if dtype not in {'float32', 'bfloat16'}:
            raise ValueError('Unsupported decoder precision')
        self.model.text_decoder.to(dtype=getattr(torch, dtype))
        self.model.shared.to(dtype=getattr(torch, dtype))
        self.model.lm_head.to(dtype=getattr(torch, dtype))
        self.provenance['decoder_dtype'] = dtype

    def configure_adapters(self, rank: int = 4) -> list[nn.Parameter]:
        self.requires_grad_(False)
        # Shared pretrained tensors are never copied or optimized. Attention
        # adapters have independent parameters and manageable optimizer states.
        for layer in self.model.text_decoder.layers:
            for attention in (layer.self_attn, layer.cross_attention.original):
                for name in ("q_proj", "v_proj"):
                    module = getattr(attention, name)
                    if isinstance(module, LoRALinear):
                        if module.rank != rank:
                            raise ValueError("Cannot change the rank of installed adapters")
                        module.A.requires_grad_(True)
                        module.B.requires_grad_(True)
                    else:
                        setattr(attention, name, FloatAdapterLinear(module, rank))
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_adapter_state(self, state: dict[str, Tensor]):
        expected = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if expected.keys() != state.keys():
            raise ValueError("Adapter snapshot names differ from configured trainable parameters")
        with torch.no_grad():
            for name, parameter in expected.items():
                if parameter.shape != state[name].shape:
                    raise ValueError(f"Adapter snapshot shape mismatch: {name}")
                parameter.copy_(state[name])

    def _suppress_noncontent_tokens(self, scores):
        for token in self.tokenizer.all_special_ids:
            if token != self.eos_id:
                scores[..., token] = -torch.inf
        return scores

    def next_token(self, states, token_ids, wait_k, encoder_length=None):
        if states.ndim != 2 or not len(states):
            raise ValueError("next_token needs nonempty observed [chunks, dimension] prefix")
        previous = torch.tensor([[self.bos_id, *token_ids]], device=states.device)
        lengths = torch.tensor([len(states)], device=states.device)
        enc_len = None if encoder_length is None else lengths.new_tensor([encoder_length])
        logits = self.logits(states[None], lengths, previous, wait_k, enc_len)[0, -1]
        return int(self._suppress_noncontent_tokens(logits).argmax())

    def _decode_hidden(self, states, lengths, input_ids, wait_k, encoder_lengths=None,
                       past_key_values=None, use_cache=False):
        if wait_k < 1:
            raise ValueError("wait_k must be positive")
        b, t, d = states.shape
        if d != self.state_dim:
            raise ValueError("Incorrect committed source state dimension")
        flat = states.reshape(b, t * self.pre_decision_ratio, self.encoder_dim).to(self.model_dtype)
        real_lengths = lengths * self.pre_decision_ratio if encoder_lengths is None else encoder_lengths
        source_mask = torch.arange(flat.shape[1], device=flat.device)[None] < real_lengths[:, None]
        if past_key_values is None:
            position_offset = 0
        elif hasattr(past_key_values, "get_seq_length"):
            position_offset = int(past_key_values.get_seq_length())
        else:
            position_offset = int(past_key_values[0][0].shape[2])
        for layer in self.model.text_decoder.layers:
            layer.cross_attention.wait_k = wait_k
            layer.cross_attention.position_offset = position_offset
        return self.model.text_decoder(input_ids=input_ids, encoder_hidden_states=flat,
                                       encoder_attention_mask=source_mask,
                                       past_key_values=past_key_values, use_cache=use_cache,
                                       return_dict=True)

    def logits(self, states, lengths, previous_tokens, wait_k, encoder_lengths=None):
        # Generic TranslationBackend uses [BOS, previous lexical tokens]. The
        # pretrained Arabic decoder needs the compulsory language token too.
        prompt = previous_tokens.new_full((len(previous_tokens), 1), self.language_id)
        input_ids = torch.cat((previous_tokens[:, :1], prompt, previous_tokens[:, 1:]), dim=1)
        hidden = self._decode_hidden(states, lengths, input_ids, wait_k, encoder_lengths)
        return self.model.lm_head(hidden.last_hidden_state[:, 1:])

    def loss(self, states: Tensor, batch: EncodedBatch, wait_k: int) -> Tensor:
        targets = batch.target_tokens
        previous = torch.cat([targets.new_full((targets.shape[0], 1), self.bos_id), targets[:, :-1]], dim=1)
        language = previous.new_full((len(previous), 1), self.language_id)
        input_ids = torch.cat((previous[:, :1], language, previous[:, 1:]), dim=1)
        hidden = self._decode_hidden(states, batch.lengths, input_ids, wait_k, batch.encoder_lengths)
        valid = torch.arange(targets.shape[1], device=targets.device)[None] < batch.target_lengths[:, None]
        # Decoder attention is unchanged. Padding has zero loss/gradient, so
        # omit only its expensive 256102-way output projection. This preserves
        # each utterance's token-average CE and all mathematically nonzero
        # source/adapter derivatives while avoiding a padded [B,U,V] tensor.
        logits = self.model.lm_head(hidden.last_hidden_state[:, 1:][valid])
        token_loss = F.cross_entropy(logits.float(), targets[valid],
                                     reduction="none", ignore_index=self.pad_id)
        row = torch.arange(len(targets), device=targets.device)[:, None].expand_as(targets)[valid]
        losses = token_loss.new_zeros(len(targets)).scatter_add(0, row, token_loss)
        return losses / batch.target_lengths

    @torch.no_grad()
    def encode_waveform(self, waveform: Tensor, sample_rate: int = 16000) -> tuple[Tensor, Tensor, int]:
        if sample_rate != 16000 or waveform.ndim != 1 or waveform.numel() < 1:
            raise ValueError("Expected nonempty mono 16000Hz waveform")
        chunk_samples = self.chunk_ms * sample_rate // 1000
        context_ms = getattr(self, "encoder_context_ms", None)
        context_chunks = None if context_ms is None else context_ms // self.chunk_ms
        blocks, ends, committed = [], [], 0
        raw = waveform.detach().cpu().float().numpy()
        for chunk_index, end in enumerate(range(chunk_samples, len(raw) + chunk_samples, chunk_samples)):
            end = min(end, len(raw))
            # Critically, feature extraction AND its normalization receive only
            # observed audio. In bounded mode the left edge advances only at
            # chunk boundaries, preserving the 160ms subsampling phase.
            start = (0 if context_chunks is None else
                     max(0, (chunk_index + 1 - context_chunks) * chunk_samples))
            observed = raw[start:end]
            if len(observed) < 560:
                # Only an observed source EOS can expose a chunk this short.
                # Two 25ms windows at 10ms stride avoid undefined one-frame
                # sample variance in the checkpoint's CMVN. These are known
                # synthetic zeros, never unavailable future audio samples.
                observed = F.pad(waveform.detach().cpu().float()[:end], (0, 560 - end)).numpy()
            features = self.processor.feature_extractor(observed, sampling_rate=sample_rate, return_tensors="pt")
            output = self.model.speech_encoder(
                input_features=features.input_features.to(device=self.device,
                    dtype=next(self.model.speech_encoder.parameters()).dtype),
                attention_mask=features.attention_mask.to(self.device), return_dict=True)
            real_length = int(self.model._compute_sub_sample_lengths_from_attention_mask(features.attention_mask)[0])
            encoded = output.last_hidden_state[0, :real_length]
            # Once the window slides, its first W-1 complete source chunks
            # account for (W-1)*ratio local encoder states. The remaining
            # states are exactly the newly exposed block, including a partial
            # observed-EOS block. ``committed`` remains the global count.
            local_previous = (committed if start == 0 else
                              (context_chunks - 1) * self.pre_decision_ratio)
            count = len(encoded) - local_previous
            if count == 0 and end == len(raw) and blocks:
                break  # observed EOS creates no additional speech state
            if count <= 0 or count > self.pre_decision_ratio:
                raise RuntimeError(f"Unsupported source subsampling at {end}: {count} new states")
            if end < len(raw) and count != self.pre_decision_ratio:
                raise RuntimeError("A complete chunk did not expose the expected speech states")
            block = encoded[-count:]
            blocks.append(F.pad(block, (0, 0, 0, self.pre_decision_ratio - count)).flatten().float())
            ends.append(end * 1000 / sample_rate)
            committed += count
        return torch.stack(blocks), torch.tensor(ends, device=self.device), committed

    @torch.no_grad()
    def encode(self, manifest: str | Path | list[dict[str, Any]], limit: int | None = None) -> EncodedBatch:
        records = read_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            records = records[:limit]
        if not records:
            raise ValueError("No records to encode")
        states, times, targets, durations, enc_lengths = [], [], [], [], []
        for record in records:
            audio, rate = read_audio(record)
            x, ends, n = self.encode_waveform(audio, rate)
            ids = self.tokenizer.encode(record["target_text"], add_special_tokens=False)
            ids.append(self.eos_id)
            states.append(x); times.append(ends)
            targets.append(torch.tensor(ids, device=self.device, dtype=torch.long))
            durations.append(audio.numel() * 1000 / rate); enc_lengths.append(n)
        lengths = torch.tensor([len(x) for x in states], device=self.device)
        padded_times = nn.utils.rnn.pad_sequence(times, batch_first=True)
        for i, n in enumerate(lengths.tolist()):
            padded_times[i, n:] = times[i][-1]
        return EncodedBatch(
            nn.utils.rnn.pad_sequence(states, batch_first=True), lengths,
            nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_id),
            torch.tensor([len(y) for y in targets], device=self.device), padded_times,
            torch.tensor(durations, device=self.device), [r["target_text"] for r in records],
            [r["id"] for r in records], torch.tensor(enc_lengths, device=self.device),
            {**self.provenance, "corpus": ",".join(sorted({r.get("corpus", "supplied_audio") for r in records})),
             "splits": sorted({r.get("split", "unspecified") for r in records})})

    def detokenize(self, ids, delays, terminal_delay):
        pieces = self.tokenizer.convert_ids_to_tokens(ids)
        word_delays, pending = [], False
        for piece, delay in zip(pieces, delays):
            if piece.startswith("▁") and pending:
                word_delays.append(float(delay))
                pending = False
            if piece.replace("▁", ""):
                pending = True
        if pending:
            word_delays.append(float(terminal_delay))
        text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        # SentencePiece's output may contain a standalone whitespace marker.
        # The timing parser tracks nonempty pending words, matching text.split.
        if len(word_delays) != len(text.split()):
            raise RuntimeError("SentencePiece word/timing alignment failed")
        return text, word_delays

    @torch.no_grad()
    def decode(self, states: Tensor, batch: EncodedBatch, wait_k: int, max_new_tokens: int = 128) -> list[Generation]:
        """Batched greedy generation, cached causal history, reference-free cap."""
        if wait_k < 1 or max_new_tokens <= 0:
            raise ValueError("wait_k and max_new_tokens must be positive")
        b = len(states)
        input_ids = torch.tensor([[self.bos_id, self.language_id]] * b, device=states.device)
        cache = None
        tokens, delays = [[] for _ in range(b)], [[] for _ in range(b)]
        finished, eos_delays = [False] * b, [None] * b
        for m in range(max_new_tokens):
            hidden = self._decode_hidden(states, batch.lengths, input_ids, wait_k,
                                         batch.encoder_lengths, cache, use_cache=True)
            cache = hidden.past_key_values
            scores = self.model.lm_head(hidden.last_hidden_state[:, -1])
            # No forced content, reference lengths, or target-language alphabet
            # restriction is used. Only non-content special tokens are blocked.
            self._suppress_noncontent_tokens(scores)
            chosen = scores.argmax(-1)
            for i in range(b):
                if finished[i]:
                    continue
                length = int(batch.lengths[i])
                read = min(length, wait_k + m)
                delay = (float(batch.durations_ms[i]) if wait_k + m > length
                         else float(batch.chunk_end_ms[i, read - 1]))
                token = int(chosen[i])
                if token == self.eos_id:
                    finished[i], eos_delays[i] = True, delay
                else:
                    tokens[i].append(token); delays[i].append(delay)
            if all(finished):
                break
            input_ids = chosen[:, None]
        generations = []
        for i in range(b):
            terminal = (eos_delays[i] if eos_delays[i] is not None else
                        delays[i][-1] if delays[i] else float(batch.durations_ms[i]))
            text, word_times = self.detokenize(tokens[i], delays[i], terminal)
            generations.append(Generation(text, tokens[i], delays[i], word_times, finished[i], eos_delays[i]))
        return generations
