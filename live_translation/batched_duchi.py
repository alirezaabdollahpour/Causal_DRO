"""Vectorized nested Causal Duchi for one branching point per prefix.

Training-neighbor scenarios share the observed current prefix. With distinct
next source vectors their remaining paths are deterministic, so all child
recourse problems can be solved in parallel. Every leaf has its own Armijo
line search; unrelated utterances never share an acceptance decision.
Unusual duplicate-prefix trees use the general nested implementation.
"""
from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import torch
from torch import Tensor

from .attacks import AttackConfig, NestedSolveBudgetExceeded, ScenarioTree, causal_duchi_step, prefix_node_ids, transport_cost
from .data import EncodedBatch


def _selected_scenarios(prefix: Tensor, bank: EncodedBatch, count: int) -> EncodedBatch:
    """Only observed source vectors enter donor selection or splicing."""
    t = len(prefix)
    eligible = torch.nonzero(bank.lengths >= t).flatten()
    if not len(eligible):
        raise ValueError("Training continuation bank has no source this long")
    distance = (bank.states[eligible, :t] - prefix.to(bank.states.device)[None]).square().mean((1, 2))
    nearest = eligible[torch.argsort(distance, stable=True)[:count]]
    selected = bank.subset(nearest).to(prefix.device)
    x = selected.states.detach().clone()
    x[:, :t] = prefix[None]
    return replace(selected, states=x)


def _one_branch(scenarios: EncodedBatch, t: int, count: int) -> bool:
    if len(scenarios.ids) != count or count < 2:
        return False
    survivors = torch.nonzero(scenarios.lengths > t + 1).flatten().tolist()
    # With no child, the current node is itself a deterministic suffix. The
    # general solver applies its line search to that shared current action.
    if not survivors:
        return False
    for index, left in enumerate(survivors):
        for right in survivors[index + 1:]:
            if torch.equal(scenarios.states[left, t + 1], scenarios.states[right, t + 1]):
                return False
    return True


def _concatenate(items: list[EncodedBatch]) -> EncodedBatch:
    payload = {}
    for field in fields(EncodedBatch):
        values = [getattr(item, field.name) for item in items]
        if isinstance(values[0], Tensor):
            payload[field.name] = torch.cat(values, dim=0)
        elif field.name in {"ids", "references"}:
            payload[field.name] = sum(values, [])
        else:
            payload[field.name] = None
    return _trim_scenarios(EncodedBatch(**payload))


def _trim_scenarios(scenarios: EncodedBatch) -> EncodedBatch:
    """Discard padding inherited from unselected members of the full bank."""
    source_length = int(scenarios.lengths.max())
    target_length = int(scenarios.target_lengths.max())
    return replace(scenarios, states=scenarios.states[:, :source_length],
                   chunk_end_ms=scenarios.chunk_end_ms[:, :source_length],
                   target_tokens=scenarios.target_tokens[:, :target_length])


def _parallel_suffixes(x: Tensor, valid: Tensor, initial: Tensor, free: Tensor,
                       loss_fn, config: AttackConfig) -> tuple[Tensor, dict[str, Any]]:
    """Exact vectorization of independent deterministic _shared_ascent calls."""
    action = initial.detach().clone()
    enabled = free.any(1)
    updates = torch.zeros(len(x), dtype=torch.long, device=x.device)
    failed = torch.zeros_like(enabled)
    calls = 0
    for _ in range(config.steps):
        with torch.enable_grad():
            probe = action.detach().requires_grad_(True)
            losses = loss_fn(probe)
            calls += 1
            objective = losses - config.lam * transport_cost(x, probe, valid, config.cost_scale)
            gradient = torch.autograd.grad(objective.sum(), probe)[0]
            direction = gradient * free[..., None]
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError("Batched Duchi suffix direction is nonfinite")
        residual = direction.norm(dim=-1).max(-1).values
        active = enabled & (residual > config.tolerance)
        enabled = active
        if not bool(active.any()):
            break
        derivative = (gradient * direction).sum((1, 2)).detach()
        objective = objective.detach()
        steps = x.new_full((len(x),), config.initial_step)
        accepted = ~active
        candidate = action.detach().clone()
        for _ in range(config.backtracking_steps):
            pending = active & ~accepted
            if not bool(pending.any()):
                break
            proposed_y = torch.where(pending[:, None, None],
                                     action.detach() + steps[:, None, None] * direction.detach(), candidate)
            with torch.no_grad():
                proposed = loss_fn(proposed_y) - config.lam * transport_cost(x, proposed_y, valid, config.cost_scale)
                calls += 1
            slack = 8 * torch.finfo(x.dtype).eps * torch.maximum(torch.ones_like(objective), objective.abs())
            take = pending & (proposed >= objective + 1e-4 * steps * derivative - slack)
            candidate = torch.where(take[:, None, None], proposed_y, candidate)
            accepted |= take
            steps = torch.where(pending & ~take, steps * .5, steps)
        just_failed = active & ~accepted
        failed |= just_failed
        enabled &= ~just_failed
        updates += (active & accepted).long()
        action = candidate.detach()
    with torch.enable_grad():
        probe = action.detach().requires_grad_(True)
        objective = loss_fn(probe) - config.lam * transport_cost(x, probe, valid, config.cost_scale)
        calls += 1
        full_gradient = torch.autograd.grad(objective.sum(), probe)[0]
        direction = full_gradient * free[..., None]
    residual = direction.norm(dim=-1).max(-1).values.detach()
    return action, {"residual": residual, "converged": residual <= config.tolerance,
                    "line_search_failed": failed, "iterations": updates, "loss_batches": calls,
                    "full_gradient": full_gradient.detach()}


def _solve_roots(scenarios: EncodedBatch, previous: Tensor, t: int, roots: int,
                 count: int, backend, k: int, config: AttackConfig,
                 max_evaluations: int, final_diagnostics: bool) -> tuple[Tensor, dict[str, Any]]:
    x, valid = scenarios.states.detach(), scenarios.mask
    current = x.reshape(roots, count, *x.shape[1:])[:, 0, t].clone()
    free = valid & (torch.arange(x.shape[1], device=x.device)[None] > t)
    done = torch.zeros(roots, dtype=torch.bool, device=x.device)
    loss_batches = 0
    line_search_failed = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    updates = torch.zeros(roots, dtype=torch.long, device=x.device)
    evaluations = 0

    def counted_loss(y):
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise NestedSolveBudgetExceeded(f"Batched nested Causal Duchi exceeded max_evaluations={max_evaluations}")
        evaluations += 1
        value = backend.loss(y, scenarios, k)
        if value.shape != (len(y),) or not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Batched Duchi requires one finite, uncoupled loss per donor path")
        return value

    final_checked = False
    for iteration in range(config.steps + int(final_diagnostics)):
        # Every current candidate restarts every child from its CLEAN suffix,
        # reproducing the report's nested reoptimization convention.
        initial = x.clone()
        if t:
            initial[:, :t] = previous.repeat_interleave(count, dim=0)
        initial[:, t] = current.repeat_interleave(count, dim=0)
        continuation, child = _parallel_suffixes(x, valid, initial, free,
                                                 counted_loss, config)
        loss_batches += child["loss_batches"]
        line_search_failed |= child["line_search_failed"]
        # The child solver's final full-action derivative already holds its
        # optimized suffix fixed. Averaging its current-date component is the
        # same envelope derivative as a separate current-node backward pass.
        direction = child["full_gradient"][:, t].reshape(roots, count, -1).mean(1)
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError("Batched Duchi current direction is nonfinite")
        residual = direction.norm(dim=-1).detach()
        final_checked = True
        done |= residual <= config.tolerance
        if (final_diagnostics and iteration == config.steps) or bool(done.all()):
            break
        current = current.detach() + config.initial_step * torch.where(done[:, None], torch.zeros_like(direction), direction.detach())
        updates += (~done).long()
        final_checked = False
    child_residual = child["residual"].reshape(roots, count).max(1).values
    combined = torch.maximum(residual, child_residual)
    return current.detach(), {"max_direction_residual": float(combined.max()),
                              "pre_update_residual": float(combined.max()) if not final_checked else None,
                              "residual_location": "final_action_and_recourse" if final_checked else "pre_update",
                              "final_residual_evaluated": final_checked,
                              "final_residual_unavailable": not final_checked,
                              "converged": final_checked and bool((combined <= config.tolerance).all()),
                              "line_search_failed": bool(line_search_failed.any()),
                              "root_updates": int(updates.sum()), "loss_batches": loss_batches,
                              "objective_evaluations": evaluations, "max_evaluations": max_evaluations,
                              "roots": roots, "scenario_paths": len(x)}


def batched_streaming_duchi(batch: EncodedBatch, bank: EncodedBatch, backend,
                            k: int, config: AttackConfig, count: int = 2,
                            max_evaluations: int = 10000,
                            final_diagnostics: bool = True) -> tuple[Tensor, dict[str, Any]]:
    """Commit causal attacks in parallel while using only training donors.

    ``batch.target_tokens/references/durations_ms/ids`` are never used by this
    attack. Source validity determines which externally arriving streams are
    active, but cannot change an individual row's action. The per-row loss
    returned by ``backend.loss`` must not couple different batch members.

    ``final_diagnostics=False`` skips recourse recomputation after the LAST
    applied current-node update. That solve cannot affect the committed action;
    its omission preserves the policy but leaves final stationarity unmeasured.
    Recourse is still recomputed before EVERY applied current-node gradient.
    """
    if count < 1 or k < 1 or max_evaluations < 1:
        raise ValueError("Invalid conditional sampling or solver configuration")
    if batch.states.shape[-1] != bank.states.shape[-1]:
        raise ValueError("Source and training donor dimensions differ")
    output = batch.states.detach().clone()
    diagnostics = []
    fallbacks, vectorized_roots = 0, 0
    for t in range(int(batch.lengths.max())):
        rows, selected = [], []
        for i in torch.nonzero(batch.lengths > t).flatten().tolist():
            prefix = batch.states[i, :t + 1]
            scenarios = _selected_scenarios(prefix, bank, count)
            if _one_branch(scenarios, t, count):
                rows.append(i); selected.append(scenarios)
            else:
                scenarios = _trim_scenarios(scenarios)
                # The callback closes over training scenarios only; it does not
                # capture this utterance's held-out reference or future source.
                def sampler(observed, scenarios=scenarios):
                    if not torch.equal(scenarios.states[:, :len(observed)], observed[None].expand(len(scenarios.ids), -1, -1)):
                        raise ValueError("Fallback scenario prefix mismatch")
                    tree = ScenarioTree(scenarios.states, scenarios.mask,
                                        prefix_node_ids(scenarios.states, scenarios.mask))
                    return tree, lambda y: backend.loss(y, scenarios, k)
                action, diagnostic = causal_duchi_step(prefix, output[i, :t], sampler, config,
                                                       solver="nested", max_evaluations=max_evaluations)
                diagnostic.update(residual_location="final_action_and_recourse",
                                  final_residual_evaluated=True, final_residual_unavailable=False)
                output[i, t] = action
                diagnostics.append(diagnostic)
                fallbacks += 1
        if rows:
            scenarios = _concatenate(selected)
            actions, diagnostic = _solve_roots(scenarios, output[rows, :t], t,
                                               len(rows), count, backend, k, config, max_evaluations,
                                               final_diagnostics)
            output[rows, t] = actions
            diagnostics.append(diagnostic)
            vectorized_roots += len(rows)
    final_evaluated = all(d["final_residual_evaluated"] for d in diagnostics)
    return output, {
        "method": "causal_duchi_batched_nested_conditional_ascent",
        "conditional_model": "training_prefix_nearest_neighbor_splice",
        "solver": "nested", "continuation_solution": "resolved_before_each_current_node_gradient",
        "envelope_derivative": "descendant_actions_detached",
        "line_search": "independent_per_deterministic_leaf",
        "vectorized_roots": vectorized_roots, "general_tree_fallbacks": fallbacks,
        "steps_committed": int(batch.lengths.sum()),
        "max_direction_residual": max(d["max_direction_residual"] for d in diagnostics),
        "residual_location": "final_action_and_recourse" if final_evaluated else "pre_update_or_final_mixed",
        "final_residual_evaluated": final_evaluated,
        "final_residual_unavailable": not final_evaluated,
        "final_diagnostics_requested": bool(final_diagnostics),
        "converged": final_evaluated and all(d["converged"] for d in diagnostics),
        "exact_solve": False, "global_optimum_certified": False,
        "nodes": diagnostics,
    }
