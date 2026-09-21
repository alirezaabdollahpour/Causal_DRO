"""Exercise the real pretrained speech backend on synthetic audio, not MuST-C.

This verifies tensor/autograd and streaming API behavior, never translation
quality. The output explicitly records its synthetic provenance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import torch

from .backend import FairseqBackend
from .metrics import latency_scores


def check_attacks(backend, batch, alternative_states):
    """Exercise all attack families on a two-leaf artificial speech law.

    Its two real encoder paths share their first two chunks. References are
    arbitrary, and one PICNN update is an autograd check, not robust training.
    """
    from .attacks import (AttackConfig, PICNNAttacker, ScenarioTree,
        anticipative_attack, nested_causal_duchi_attack, prefix_node_ids,
        train_picnn_step, transport_cost)

    pair = batch.subset([0, 0])
    pair.states[1] = alternative_states
    pair.ids = ["synthetic_waveform:branch0", "synthetic_waveform:branch1"]
    pair.references[1] = "Vielen Dank ."
    pieces = " ".join(backend.spm.encode(pair.references[1], out_type=str))
    second_target = backend.dictionary.encode_line(pieces, add_if_not_exist=False, append_eos=True).long().to(pair.states.device)
    first_target = batch.target_tokens[0, :int(batch.target_lengths[0])]
    pair.target_tokens = torch.nn.utils.rnn.pad_sequence([first_target, second_target], batch_first=True, padding_value=backend.pad_id)
    pair.target_lengths = pair.target_lengths.new_tensor([len(first_target), len(second_target)])
    assert torch.equal(pair.states[0, :2], pair.states[1, :2])
    backend.requires_grad_(False).eval()
    loss_fn = lambda y: backend.loss(y, pair, wait_k=1)
    baseline_ce = loss_fn(pair.states).detach()
    config = AttackConfig(lam=1.0, steps=2, step_size=.1, tolerance=1e-5)

    direct = anticipative_attack(pair.states, pair.mask, loss_fn, config)
    assert bool(torch.isfinite(direct.y).all()) and bool((direct.costs > 0).all())
    assert float((direct.losses - config.lam * direct.costs).mean()) >= float(baseline_ce.mean()) - 1e-6
    diagnostics = {"anticipative_direct": dict(direct.diagnostics,
        output_change_max_abs=float((direct.y - pair.states).abs().max()))}

    for threat in ("causal", "anticipative"):
        # Matched initial weights; only the context information differs.
        torch.manual_seed(2031)
        policy = PICNNAttacker(backend.state_dim, context_dim=8, hidden_dim=8,
            depth=1, threat=threat, solver_steps=2, solver_tolerance=1e-4)
        before = {name: p.detach().clone() for name, p in policy.named_parameters()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        training = train_picnn_step(policy, optimizer, pair.states, pair.mask, loss_fn, config)
        gradient_norm = sum(float(p.grad.square().sum()) for p in policy.parameters() if p.grad is not None) ** .5
        parameter_change = sum(float((p.detach() - before[name]).square().sum()) for name, p in policy.named_parameters()) ** .5
        assert gradient_norm > 0 and parameter_change > 0
        policy.requires_grad_(False).eval()
        y, frozen = policy(pair.states, pair.mask, config.lam, config.cost_scale)
        change = float((y - pair.states).abs().max())
        assert bool(torch.isfinite(y).all()) and change > 0
        detail = dict(training_updates=1, training_gradient_norm=gradient_norm,
            parameter_change_l2=parameter_change, output_change_max_abs=change,
            synthetic_mean_ce=float(loss_fn(y).detach().mean()),
            mean_transport_cost=float(transport_cost(pair.states, y, pair.mask).mean()),
            training_diagnostics=training, frozen_diagnostics=frozen)
        if threat == "causal":
            torch.testing.assert_close(y[0, :2], y[1, :2], rtol=1e-5, atol=1e-5)
            prefix_y, _ = policy(pair.states[:, :2], pair.mask[:, :2], config.lam, config.cost_scale)
            torch.testing.assert_close(y[:, :2], prefix_y, rtol=1e-5, atol=1e-5)
            detail.update(frozen_suffix_intervention_prefix_invariant=True,
                          frozen_full_vs_prefix_call_invariant=True)
        diagnostics[threat + "_picnn"] = detail

    tree = ScenarioTree(pair.states, pair.mask, prefix_node_ids(pair.states, pair.mask))
    nested_config = AttackConfig(lam=1.0, steps=1, step_size=.1, tolerance=1e-5)
    nested = nested_causal_duchi_attack(tree, loss_fn, nested_config, max_evaluations=256)
    assert bool(torch.isfinite(nested.y).all()) and bool((nested.costs > 0).all())
    torch.testing.assert_close(nested.y[0, :2], nested.y[1, :2], rtol=0, atol=0)
    diagnostics["causal_duchi_nested"] = dict(nested.diagnostics,
        shared_prefix_actions_exactly_equal=True,
        output_change_max_abs=float((nested.y - pair.states).abs().max()))
    return dict(status="passed", corpus="two_synthetic_waveforms", scenario_paths=2,
        shared_initial_chunks=2, arbitrary_references=pair.references,
        source_law="two explicit equiprobable paths; no learned corpus continuation model",
        clean_synthetic_ce=baseline_ce.tolist(), diagnostics=diagnostics,
        interpretation="Numerical execution and information-restriction audit only; no benchmark robustness result or optimality certificate")


def check(checkpoint, data_bin, user_dir, output):
    import soundfile as sf
    from simuleval.agents.actions import WriteAction
    from simuleval.data.segments import SpeechSegment
    from simuleval.evaluator.scorers.latency_scorer import ALScorer, APScorer, DALScorer, LAALScorer
    from .simuleval_agent import CausalRobustSpeechAgent

    torch.set_num_threads(2)
    torch.manual_seed(2027)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    waveform = .1 * torch.sin(torch.arange(19200).float() * .07)  # 1.2 seconds
    audio = output / "synthetic_sine.wav"
    sf.write(audio, waveform.numpy(), 16000, subtype="FLOAT")
    record = dict(id="synthetic_waveform:0", audio=str(audio.resolve()), target_text="Guten Morgen Welt .",
                  split="integration_audit", corpus="synthetic_waveform")
    backend = FairseqBackend(checkpoint, data_bin, user_dir)
    batch = backend.encode([record])
    assert batch.metadata["corpus"] == "synthetic_waveform"

    perturbed_audio = waveform.clone()
    perturbed_audio[9200:] = -.2 * torch.cos(torch.arange(len(waveform) - 9200).float() * .13)
    changed, _, _ = backend.encode_waveform(perturbed_audio)
    torch.testing.assert_close(batch.states[0, :2], changed[:2], rtol=1e-5, atol=1e-5)
    state_difference_after = float((batch.states[0, 2:] - changed[2:]).abs().max())
    assert state_difference_after > 0

    adapters = backend.configure_adapters(4)
    states = batch.states.detach().requires_grad_(True)
    ce = backend.loss(states, batch, wait_k=1)
    ce.mean().backward()
    assert bool(torch.isfinite(ce).all()) and states.grad.abs().sum() > 0
    assert backend.model.decoder.output_projection.B.grad.abs().sum() > 0
    previous = batch.target_tokens.new_full((1, 1), backend.bos_id)
    first_states = batch.states.detach().requires_grad_(True)
    first_logit = backend.logits(first_states, batch.lengths, previous, 1, batch.encoder_lengths)[0, 0, 5]
    gradient, = torch.autograd.grad(first_logit, first_states)
    assert torch.count_nonzero(gradient[:, 1:]) == 0

    generation = backend.decode(batch.states, batch, wait_k=1, max_new_tokens=8)[0]
    args = SimpleNamespace(live_checkpoint=checkpoint, live_data_bin=data_bin, live_user_dir=user_dir,
        live_config_yaml="config_st.yaml", live_device="cpu", live_wait_k=1, live_max_new_tokens=8,
        live_adapter_rank=4, live_defender_state=None, live_picnn_state=None, live_duchi_state=None, live_attack_factory=None)
    agent = CausalRobustSpeechAgent(args)
    words, word_times, done = [], [], False
    received = 0
    while not done:
        if received < len(waveform):
            end = min(received + 80, len(waveform))  # actual 5ms incremental pushes
            agent.push(SpeechSegment(content=waveform[received:end].tolist(), sample_rate=16000, finished=end == len(waveform)))
            received = end
        action = agent.policy()
        while isinstance(action, WriteAction):
            if action.content:
                for word in action.content.split():
                    words.append(word)
                    word_times.append(received / 16.)
            if action.finished:
                done = True
                break
            action = agent.policy()
        if received == len(waveform) and not done:
            raise AssertionError("Agent requested more audio after source EOS")
    assert " ".join(words) == generation.text
    assert word_times == generation.word_delays_ms
    assert agent.truncated == (not generation.finished)

    attack_checks = check_attacks(backend, batch, changed)

    latency_fixtures = []
    for delays, ref in [([200, 400, 600, 800, 1000], 5), ([500, 1000, 1000, 1000], 2)]:
        instance = SimpleNamespace(delays=delays, source_length=1000, reference="fixture", reference_length=ref)
        expected = {cls.__name__.replace("Scorer", ""): cls().compute(instance) for cls in (ALScorer, APScorer, DALScorer, LAALScorer)}
        actual = latency_scores(delays, 1000, ref)
        assert all(abs(expected[k] - actual[k]) < 1e-8 for k in expected)
        latency_fixtures.append(dict(delays=delays, reference_words=ref, upstream=expected, actual=actual))

    result = dict(status="passed", benchmark_result=False, corpus="synthetic_waveform",
        source_duration_ms=1200, arbitrary_reference=record["target_text"], backend=backend.provenance,
        encoded_shape=list(batch.states.shape), encoder_lengths=batch.encoder_lengths.tolist(),
        chunk_end_ms=batch.chunk_end_ms[0].tolist(), synthetic_ce=float(ce.detach().mean()),
        source_gradient_norm=float(states.grad.norm()), trainable_adapter_parameters=sum(p.numel() for p in adapters),
        source_prefix_invariance=True, later_state_intervention_max_abs=state_difference_after,
        first_token_future_source_gradient_zero=True, standalone_generation=asdict(generation),
        simuleval_incremental_text=" ".join(words), simuleval_incremental_word_times_ms=word_times,
        incremental_matches_standalone=True, latency_matches_simuleval_1_1_4=latency_fixtures,
        attack_integration=attack_checks,
        environment={"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
             ("torch", "torchaudio", "fairseq", "numpy", "omegaconf", "hydra-core", "sentencepiece", "simuleval", "sacrebleu")}},
        limitations=["Synthetic sine wave and arbitrary German reference; CE is only a numerical integration probe.",
                      "Attacks use two artificial encoder paths and arbitrary references; PICNN fitting is exactly one numerical test update.",
                      "No MuST-C corpus, corpus-trained speech attack, translation-quality result, or GPU throughput measurement.",
                      "Standalone and incremental delays exclude computation time."])
    (output / "integration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-bin", required=True)
    parser.add_argument("--user-dir", required=True)
    parser.add_argument("--output", default="live_translation/runs/backend_check")
    args = parser.parse_args()
    check(args.checkpoint, args.data_bin, args.user_dir, args.output)


if __name__ == "__main__":
    main()
