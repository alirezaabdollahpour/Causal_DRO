"""Attacks on speech representations with controls on what they can observe.

``loss_fn`` returns one loss per utterance. Causal attacks choose each action
using only audio observed so far. A direct causal attack uses a tree of
possible continuations, where utterances with the same prefix share an action.
Learned attackers are fitted on training data before test evaluation.

Transport cost sums squared changes to observed chunks and divides by one
fixed scale. It does not depend on the utterance's eventual length.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

LossFunction = Callable[[Tensor], Tensor]


def _validate_paths(x: Tensor, valid: Tensor) -> None:
    if x.ndim != 3 or valid.shape != x.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("expected x [N,T,D] and boolean valid [N,T]")
    if min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("paths must have nonempty floating-point dimensions")
    if valid.device != x.device:
        raise ValueError("paths and validity mask must be on the same device")
    if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("validity masks must describe nonempty prefixes")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("source paths, including padding, must be finite")


def transport_cost(x: Tensor, y: Tensor, valid: Tensor, cost_scale: float = 1.0) -> Tensor:
    """Per-path additive squared cost, in a fixed common system of units."""
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("cost_scale must be a fixed positive finite scalar")
    if y.shape != x.shape or valid.shape != x.shape[:2]:
        raise ValueError("incompatible path shapes")
    return ((y - x).square().sum(-1) * valid).sum(-1) / cost_scale


def _probabilities(x: Tensor, probabilities: Tensor | None) -> Tensor:
    if probabilities is None:
        return x.new_full((x.shape[0],), 1.0 / x.shape[0])
    p = torch.as_tensor(probabilities, dtype=x.dtype, device=x.device).detach()
    if p.shape != (x.shape[0],) or not bool(torch.isfinite(p).all()) or not bool((p > 0).all()):
        raise ValueError("probabilities must be finite, positive, and one per path")
    p = p / p.max()
    p = p / p.sum()
    if not bool((p > 0).all()):
        raise ValueError("probability dynamic range exceeds the path dtype")
    return p


def _losses(loss_fn: LossFunction, y: Tensor) -> Tensor:
    value = loss_fn(y)
    if value.shape != (y.shape[0],):
        raise ValueError("loss_fn must return one scalar per path, shape [N]")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("attack loss became nonfinite")
    return value


@dataclass(frozen=True)
class AttackConfig:
    lam: float = 1.0
    steps: int = 100
    step_size: float | None = None
    tolerance: float = 1e-6
    cost_scale: float = 1.0
    backtracking_steps: int = 16

    def __post_init__(self) -> None:
        for name in ("lam", "tolerance", "cost_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.steps < 1 or self.backtracking_steps < 1:
            raise ValueError("iteration counts must be positive")
        if self.step_size is not None and (not math.isfinite(self.step_size) or self.step_size <= 0):
            raise ValueError("step_size must be finite and positive")

    @property
    def initial_step(self) -> float:
        return self.cost_scale / (2.0 * self.lam) if self.step_size is None else self.step_size


@dataclass
class AttackResult:
    y: Tensor
    losses: Tensor
    costs: Tensor
    diagnostics: dict[str, Any]


@dataclass
class ScenarioTree:
    """Finite adapted source law, with identical actions at identical prefixes.

    ``node_ids`` are local to a time column; -1 denotes padding. Two paths may
    share an ID only if their entire observed source prefixes coincide. Nodes
    may split and may never merge. IDs must also identify *all* equal prefixes
    within the supplied tree, so a leaf index cannot disguise future knowledge.
    References belong in ``loss_fn``; they are never part of a prefix key.
    """

    x: Tensor
    valid: Tensor
    node_ids: Tensor
    probabilities: Tensor | None = None

    def __post_init__(self) -> None:
        _validate_paths(self.x, self.valid)
        if self.node_ids.shape != self.valid.shape or self.node_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("node_ids must be an integer tensor shaped [N,T]")
        self.node_ids = self.node_ids.to(self.x.device)
        if not bool((self.node_ids[~self.valid] == -1).all()) or bool((self.node_ids[self.valid] < 0).any()):
            raise ValueError("valid nodes need nonnegative IDs; padding needs ID -1")
        self.probabilities = _probabilities(self.x, self.probabilities)
        for t in range(self.x.shape[1]):
            rows = torch.where(self.valid[:, t])[0]
            # Paths share an action only when their observed prefixes match
            # exactly. Grouping merely similar prefixes would change what the
            # attacker is allowed to know.
            prefix_to_node: dict[bytes, int] = {}
            node_to_prefix: dict[int, bytes] = {}
            for i in rows.tolist():
                node = int(self.node_ids[i, t])
                prefix = self.x[i, :t + 1].detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
                # Signed zero denotes the same observation.
                if bool((self.x[i, :t + 1] == 0).any()):
                    canonical = self.x[i, :t + 1].detach().clone()
                    canonical[canonical == 0] = 0
                    prefix = canonical.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                if node in node_to_prefix and node_to_prefix[node] != prefix:
                    raise ValueError("a scenario node joins different observed source prefixes")
                if prefix in prefix_to_node and prefix_to_node[prefix] != node:
                    raise ValueError("identical source prefixes must share one scenario node")
                node_to_prefix[node] = prefix
                prefix_to_node[prefix] = node


def prefix_node_ids(x: Tensor, valid: Tensor) -> Tensor:
    """Build exact-prefix tree IDs; singleton empirical paths remain singletons.

    This utility does not estimate a conditional law. With unique early
    prefixes an empirical tree has no future uncertainty, a limitation callers
    must report rather than interpreting it as population causal robustness.
    """
    _validate_paths(x, valid)
    ids = torch.full(valid.shape, -1, dtype=torch.long, device=x.device)
    for t in range(x.shape[1]):
        mapping: dict[tuple[float, ...], int] = {}
        for i in torch.where(valid[:, t])[0].tolist():
            key = tuple(x[i, :t + 1].detach().cpu().reshape(-1).tolist())
            ids[i, t] = mapping.setdefault(key, len(mapping))
    return ids


def _result(x: Tensor, valid: Tensor, y: Tensor, loss_fn: LossFunction,
            config: AttackConfig, probabilities: Tensor, diagnostics: dict[str, Any],
            *, differentiable: bool = False) -> AttackResult:
    losses = _losses(loss_fn, y)
    costs = transport_cost(x, y, valid, config.cost_scale)
    diagnostics.update(
        expected_loss=float((probabilities * losses).sum().detach()),
        expected_cost=float((probabilities * costs).sum().detach()),
        objective=float((probabilities * (losses - config.lam * costs)).sum().detach()),
        lambda_penalty=config.lam, cost_scale=config.cost_scale,
    )
    if not differentiable:
        y, losses, costs = y.detach(), losses.detach(), costs.detach()
    return AttackResult(y, losses, costs, diagnostics)


def _shared_ascent(initial: Tensor, masses: Tensor, expand: Callable[[Tensor], Tensor],
                   x: Tensor, valid: Tensor, loss_fn: LossFunction, config: AttackConfig,
                   probabilities: Tensor, free: Tensor) -> tuple[Tensor, dict[str, Any]]:
    """Mass-preconditioned gradient ascent of the finite stochastic program."""
    action = initial.detach().clone()
    updates = 0
    line_search_failed = False
    for _ in range(config.steps):
        with torch.enable_grad():
            action = action.detach().requires_grad_(True)
            y = expand(action)
            payoff = _losses(loss_fn, y) - config.lam * transport_cost(x, y, valid, config.cost_scale)
            objective = (probabilities * payoff).sum()
            gradient = torch.autograd.grad(objective, action)[0]
            direction = gradient / masses[:, None] * free[:, None]
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError("attack direction became nonfinite")
        residual = float(direction.norm(dim=-1).max().detach())
        if residual <= config.tolerance:
            break
        step = config.initial_step
        directional_derivative = float((gradient * direction).sum().detach())
        accepted = False
        for _ in range(config.backtracking_steps):
            proposal = action.detach() + step * direction.detach()
            with torch.no_grad():
                proposal_y = expand(proposal)
                proposed = (probabilities * (_losses(loss_fn, proposal_y) - config.lam *
                    transport_cost(x, proposal_y, valid, config.cost_scale))).sum()
            slack = 8 * torch.finfo(x.dtype).eps * max(1.0, abs(float(objective.detach())))
            if float(proposed) >= float(objective.detach()) + 1e-4 * step * directional_derivative - slack:
                action = proposal
                accepted = True
                updates += 1
                break
            step *= 0.5
        if not accepted:
            line_search_failed = True
            break
    with torch.enable_grad():
        probe = action.detach().requires_grad_(True)
        y = expand(probe)
        objective = (probabilities * (_losses(loss_fn, y) - config.lam *
            transport_cost(x, y, valid, config.cost_scale))).sum()
        direction = torch.autograd.grad(objective, probe)[0] / masses[:, None] * free[:, None]
    residual = float(direction.norm(dim=-1).max().detach())
    return expand(action.detach()), {
        "iterations": updates, "max_direction_residual": residual,
        "converged": residual <= config.tolerance,
        "line_search_failed": line_search_failed, "exact_solve": False,
        "global_optimum_certified": False,
    }


def anticipative_attack(x: Tensor, valid: Tensor, loss_fn: LossFunction,
                        config: AttackConfig, *, initial_y: Tensor | None = None,
                        probabilities: Tensor | None = None) -> AttackResult:
    """Full-path Euclidean penalized ascent (a local neural-loss optimizer)."""
    _validate_paths(x, valid)
    x = x.detach()
    p = _probabilities(x, probabilities)
    locations = valid.nonzero(as_tuple=True)
    if initial_y is not None and initial_y.shape != x.shape:
        raise ValueError("initial_y must have the source path shape")
    initial = x[locations] if initial_y is None else initial_y.detach()[locations]
    if not bool(torch.isfinite(initial).all()):
        raise ValueError("initial attacked actions must be finite")

    def expand(action: Tensor) -> Tensor:
        y = x.clone()
        y[locations] = action
        return y

    y, diagnostics = _shared_ascent(initial, p[locations[0]], expand, x, valid,
        loss_fn, config, p, torch.ones(len(initial), dtype=torch.bool, device=x.device))
    diagnostics["method"] = "anticipative_direct_ascent"
    return _result(x, valid, y, loss_fn, config, p, diagnostics)


def anticipative_duchi_attack(x: Tensor, valid: Tensor, loss_fn: LossFunction,
                              config: AttackConfig, **kwargs) -> AttackResult:
    """Classical full-path Duchi penalized ascent in Euclidean latent space.

    The caller determines label access through loss_fn. In the translation
    benchmark this oracle receives the full reference; that stronger information
    set is explicitly distinguished from a frozen source-only PICNN policy.
    Finite nonconcave ascent is a local numerical solve, not a global certificate.
    """
    result = anticipative_attack(x, valid, loss_fn, config, **kwargs)
    result.diagnostics.update(method="anticipative_duchi", global_optimum_certified=False)
    return result


def causal_duchi_attack(tree: ScenarioTree, loss_fn: LossFunction,
                        config: AttackConfig, *, fixed_prefix_y: Tensor | None = None) -> AttackResult:
    """Finite-tree Causal Duchi conditional-gradient approximation.

    For node v of mass p_v, the direction is p_v^{-1} times the shared-action
    objective gradient. It equals the probability-weighted conditional mean
    of the descendant loss gradients minus the node's quadratic-cost gradient.
    The downstream actions are current iterates: this is simultaneous ascent
    of the finite nonanticipative program, *not* exact nested Bellman recursion.
    Its finite stationarity residual is reported without a global certificate.
    """
    x, valid = tree.x.detach(), tree.valid
    p = tree.probabilities
    assert p is not None
    mapping = torch.zeros(valid.shape, dtype=torch.long, device=x.device)
    initial, masses = [], []
    for t in range(x.shape[1]):
        for node in torch.unique(tree.node_ids[valid[:, t], t]).tolist():
            rows = valid[:, t] & (tree.node_ids[:, t] == node)
            mapping[rows, t] = len(initial)
            initial.append(x[rows, t][0])
            masses.append(p[rows].sum())
    initial_tensor = torch.stack(initial)
    free = torch.ones(len(initial), dtype=torch.bool, device=x.device)
    if fixed_prefix_y is not None:
        if fixed_prefix_y.ndim != 3 or fixed_prefix_y.shape[0] != len(x) or fixed_prefix_y.shape[2] != x.shape[2]:
            raise ValueError("fixed_prefix_y must have shape [N,P,D]")
        prefix_length = fixed_prefix_y.shape[1]
        if prefix_length > x.shape[1] or not bool(valid[:, :prefix_length].all()):
            raise ValueError("fixed prefix cannot extend beyond a scenario path")
        if not bool(torch.isfinite(fixed_prefix_y).all()):
            raise ValueError("fixed attacked prefix must be finite")
        for t in range(prefix_length):
            for node_index in torch.unique(mapping[:, t]).tolist():
                values = fixed_prefix_y[mapping[:, t] == node_index, t]
                if not bool((values == values[0]).all()):
                    raise ValueError("fixed attacked prefix violates node sharing")
                initial_tensor[node_index] = values[0].detach()
                free[node_index] = False

    def expand(action: Tensor) -> Tensor:
        return torch.where(valid[..., None], action[mapping], x)

    y, diagnostics = _shared_ascent(initial_tensor, torch.stack(masses), expand,
        x, valid, loss_fn, config, p, free)
    diagnostics.update(method="causal_duchi_shared_scenario_ascent",
        scenario_paths=len(x), scenario_nodes=len(initial),
        shared_nodes=sum(int(((mapping == j) & valid).sum()) > 1 for j in range(len(initial))),
        conditional_law="supplied_finite_scenario_tree",
        continuation_solution="joint_iterates_not_nested_exact_Bellman")
    return _result(x, valid, y, loss_fn, config, p, diagnostics)


def _adaptive_checkpoints(steps: int) -> set[int]:
    """The nine checkpoint fractions specified in the ACD report."""
    return {math.ceil(fraction * steps) for fraction in
            (.22, .41, .57, .70, .80, .87, .93, .99, 1.0)}


def adaptive_causal_duchi_attack(tree: ScenarioTree, loss_fn: LossFunction,
                                 config: AttackConfig, *,
                                 fixed_prefix_y: Tensor | None = None,
                                 restarts: int = 3, seed: int = 0,
                                 max_evaluations: int = 10000) -> AttackResult:
    """Nodewise ACD on a frozen finite conditional source tree.

    Every trial current action re-solves descendant causal recourse on the
    same source/reference scenarios. The current gradient holds that recourse
    fixed (the finite-solve envelope approximation). A deterministic suffix
    is solved as one shared continuation vector to avoid recursive explosion.
    Its finite local ascent, like the nodewise ascent, has no global guarantee.
    """
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int)
            or max_evaluations < 1):
        raise ValueError("max_evaluations must be a positive integer")
    x, valid, p = tree.x.detach(), tree.valid, tree.probabilities
    assert p is not None
    prefix_length = 0
    if fixed_prefix_y is not None:
        if (fixed_prefix_y.ndim != 3 or fixed_prefix_y.shape[0] != len(x)
                or fixed_prefix_y.shape[2] != x.shape[2]):
            raise ValueError("fixed_prefix_y must have shape [N,P,D]")
        prefix_length = fixed_prefix_y.shape[1]
        if prefix_length > x.shape[1] or not bool(valid[:, :prefix_length].all()):
            raise ValueError("fixed prefix cannot extend beyond a scenario path")
        if not bool(torch.isfinite(fixed_prefix_y).all()):
            raise ValueError("fixed attacked prefix must be finite")
        for t in range(prefix_length):
            for node in torch.unique(tree.node_ids[:, t]).tolist():
                values = fixed_prefix_y[tree.node_ids[:, t] == node, t]
                if not bool((values == values[0]).all()):
                    raise ValueError("fixed attacked prefix violates node sharing")

    evaluations = 0
    node_solves = 0
    continuation_solves = 0
    checkpoint_count = 0
    step_halvings = 0
    identity_selections = 0
    root_nominal_value = x.new_zeros(())
    root_best_value = x.new_zeros(())
    checkpoints = _adaptive_checkpoints(config.steps)
    # A random initialization has order-one transport penalty at this node.
    start_std = math.sqrt(config.cost_scale / (config.lam * x.shape[-1]))

    def counted_loss(y: Tensor) -> Tensor:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise NestedSolveBudgetExceeded(
                f"Adaptive Causal Duchi exceeded max_evaluations={max_evaluations}")
        evaluations += 1
        return _losses(loss_fn, y)

    def deterministic_suffix(rows: Tensor, t: int, past: Tensor
                             ) -> tuple[Tensor, float, bool]:
        """Approximate recourse when all conditional source suffixes coincide."""
        nonlocal continuation_solves
        continuation_solves += 1
        first = int(rows[0])
        length = int(valid[first].sum())
        conditional_p = torch.zeros_like(p)
        conditional_p[rows] = p[rows] / p[rows].sum()

        def expand(action: Tensor) -> Tensor:
            path = x.clone()
            path[rows, :t] = past[None]
            path[rows, t:length] = action[None]
            return path

        suffix, diagnostic = _shared_ascent(
            x[first, t:length], x.new_ones(length - t), expand,
            x, valid, counted_loss, config, conditional_p,
            torch.ones(length - t, dtype=torch.bool, device=x.device))
        return suffix, diagnostic["max_direction_residual"], diagnostic["converged"]

    def is_deterministic(rows: Tensor, t: int) -> bool:
        first = int(rows[0])
        if not bool((valid[rows, t:] == valid[first, t:]).all()):
            return False
        suffix = x[first, t:][valid[first, t:]]
        return bool((x[rows, t:][valid[rows, t:]] ==
                     suffix.repeat(len(rows), 1)).all())

    def solve_node(rows: Tensor, t: int, past: Tensor
                   ) -> tuple[Tensor, float, bool]:
        nonlocal node_solves, checkpoint_count, step_halvings, identity_selections
        nonlocal root_nominal_value, root_best_value
        node_solves += 1
        first = int(rows[0])
        nominal = x[first, t].detach()
        # Repeated solves of the same descendant node use the same random
        # starts, so candidate current actions share a deterministic evaluator.
        node_seed = (seed + 1000003 * (t + 1) +
                     10007 * (int(tree.node_ids[first, t]) + 1)) % (2**63 - 1)
        generator = torch.Generator(device=x.device).manual_seed(node_seed)
        conditional_p = torch.zeros_like(p)
        conditional_p[rows] = p[rows] / p[rows].sum()

        def conditional_eval(z: Tensor
                             ) -> tuple[Tensor, Tensor, Tensor, float, bool]:
            # The nominal futures are frozen for all candidate z. Their
            # attacked continuations must be solved again for every z.
            path = x.clone()
            path[rows, :t] = past[None]
            path[rows, t] = z.detach()[None]
            child_residual, child_converged = 0.0, True
            if t + 1 < x.shape[1]:
                surviving = rows[valid[rows, t + 1]]
                for child in torch.unique(tree.node_ids[surviving, t + 1]).tolist():
                    child_rows = surviving[tree.node_ids[surviving, t + 1] == child]
                    next_past = torch.cat((past, z.detach()[None]))
                    if is_deterministic(child_rows, t + 1):
                        child_path, residual, converged = deterministic_suffix(
                            child_rows, t + 1, next_past)
                    else:
                        child_path, residual, converged = solve_node(
                            child_rows, t + 1, next_past)
                    path[child_rows] = child_path[child_rows]
                    child_residual = max(child_residual, residual)
                    child_converged = child_converged and converged
            with torch.enable_grad():
                probe = z.detach().requires_grad_(True)
                candidate = path.detach().clone()
                candidate[rows, t] = probe[None]
                # The Bellman node value charges the current and future
                # transport only. Earlier attacked actions are already fixed.
                payoff = counted_loss(candidate) - config.lam * transport_cost(
                    x[:, t:], candidate[:, t:], valid[:, t:], config.cost_scale)
                value = (conditional_p * payoff).sum()
                direction = torch.autograd.grad(value, probe)[0]
            if not bool(torch.isfinite(value)) or not bool(torch.isfinite(direction).all()):
                raise FloatingPointError("Adaptive Causal Duchi node evaluation became nonfinite")
            return path.detach(), value.detach(), direction.detach(), child_residual, child_converged

        # The identity action is evaluated with its own optimized continuation,
        # so the eventual committed action is at least as good on this model.
        best_z = nominal.clone()
        best_path, best_value, best_direction, best_child_residual, best_child_converged = (
            conditional_eval(best_z))
        initial_value = best_value.clone()
        best_restart = 0
        for restart in range(restarts):
            if restart == 0:
                z = nominal.clone()
                path, value, direction, residual, child_converged = (
                    best_path, best_value, best_direction,
                    best_child_residual, best_child_converged)
            else:
                z = nominal + start_std * torch.randn(
                    nominal.shape, dtype=x.dtype, device=x.device, generator=generator)
                path, value, direction, residual, child_converged = conditional_eval(z)
            previous = z.clone()
            local_z, local_path, local_value, local_direction = (
                z.clone(), path, value, direction)
            local_residual, local_child_converged = residual, child_converged
            if bool(value > best_value):
                best_z, best_path, best_value, best_direction = (
                    z.clone(), path, value, direction)
                best_child_residual, best_child_converged, best_restart = (
                    residual, child_converged, restart)
            checkpoint_best = local_value
            reduced_last = False
            increases = 0
            last_checkpoint = 0
            step = config.initial_step
            for iteration in range(1, config.steps + 1):
                if float(local_direction.norm()) <= config.tolerance:
                    break
                proposal = z + step * direction
                if iteration > 1:
                    proposal = z + .75 * (proposal - z) + .25 * (z - previous)
                next_path, next_value, next_direction, next_residual, next_child_converged = (
                    conditional_eval(proposal))
                increases += bool(next_value > value)
                if bool(next_value > local_value):
                    local_z, local_path, local_value, local_direction = (
                        proposal.clone(), next_path, next_value, next_direction)
                    local_residual, local_child_converged = (
                        next_residual, next_child_converged)
                if bool(next_value > best_value):
                    best_z, best_path, best_value, best_direction = (
                        proposal.clone(), next_path, next_value, next_direction)
                    best_child_residual, best_child_converged, best_restart = (
                        next_residual, next_child_converged, restart)
                previous, z, value, direction = z, proposal, next_value, next_direction
                if iteration in checkpoints:
                    checkpoint_count += 1
                    insufficient = increases < .75 * (iteration - last_checkpoint)
                    stagnant = not reduced_last and bool(local_value <= checkpoint_best)
                    reduce = insufficient or stagnant
                    if reduce:
                        step *= .5
                        z = local_z.clone()
                        previous = z.clone()
                        value, direction = local_value, local_direction
                        step_halvings += 1
                    reduced_last = reduce
                    checkpoint_best = local_value
                    last_checkpoint = iteration
                    increases = 0
        identity_selections += int(best_restart == 0 and bool(best_value == initial_value))
        if t == prefix_length:
            root_mass = p[rows].sum()
            root_nominal_value = root_nominal_value + root_mass * initial_value
            root_best_value = root_best_value + root_mass * best_value
        node_residual = float(best_direction.norm())
        return best_path, max(node_residual, best_child_residual), (
            node_residual <= config.tolerance and best_child_converged)

    y = x.clone()
    if fixed_prefix_y is not None:
        y[:, :prefix_length] = fixed_prefix_y.detach()
    residual, converged = 0.0, True
    if prefix_length < x.shape[1]:
        active = torch.where(valid[:, prefix_length])[0]
        for node in torch.unique(tree.node_ids[active, prefix_length]).tolist():
            rows = active[tree.node_ids[active, prefix_length] == node]
            path, node_residual, node_converged = solve_node(
                rows, prefix_length, y[int(rows[0]), :prefix_length].detach())
            y[rows] = path[rows]
            residual = max(residual, node_residual)
            converged = converged and node_converged
    diagnostics = {
        "method": "adaptive_causal_duchi_nodewise_apgd",
        "conditional_law": "supplied_finite_scenario_tree",
        "continuation_solution": "reoptimized_for_each_node_candidate",
        "deterministic_suffix_solver": "joint_shared_local_ascent",
        "envelope_derivative": "descendant_actions_detached",
        "identity_incumbent": True,
        "incumbent_scope": "supplied_conditional_model_only",
        "node_solves": node_solves,
        "continuation_solves": continuation_solves,
        "identity_selections": identity_selections,
        "nominal_conditional_objective": float(root_nominal_value),
        "best_conditional_objective": float(root_best_value),
        "restarts": restarts,
        "steps_per_restart": config.steps,
        "seed": seed,
        "momentum": .75,
        "scenario_paths": len(x),
        "scenario_nodes": sum(int(torch.unique(tree.node_ids[valid[:, t], t]).numel())
                              for t in range(x.shape[1])),
        "checkpoint_count": checkpoint_count,
        "step_halvings": step_halvings,
        "objective_evaluations": evaluations,
        "max_evaluations": max_evaluations,
        "max_direction_residual": residual,
        "converged": converged,
        "exact_solve": False,
        "global_optimum_certified": False,
    }
    result = _result(x, valid, y, counted_loss, config, p, diagnostics)
    result.diagnostics["objective_evaluations"] = evaluations
    return result


class NestedSolveBudgetExceeded(RuntimeError):
    """The explicitly bounded recursive Causal Duchi solve could not finish."""


def nested_causal_duchi_attack(tree: ScenarioTree, loss_fn: LossFunction,
                               config: AttackConfig, *,
                               fixed_prefix_y: Tensor | None = None,
                               max_evaluations: int = 10000) -> AttackResult:
    """Report-style nested conditional ascent on a finite scenario law.

    Before every current-node direction, downstream recourse is solved again
    conditional on that trial action. Descendant actions are then detached in
    the conditional loss gradient, as required by the envelope theorem. After
    the final current-node update, descendants are solved once more. Finite
    descendant residuals are explicit; this is not an exact Bellman oracle for
    nonconcave neural losses or when the node iteration limit is reached.

    A conditionally deterministic remaining source path has no nonanticipative
    restrictions beyond one shared sequence of actions. Such a suffix is solved
    jointly, avoiding exponential recursion after a scenario tree has split into
    singletons. Duplicate deterministic paths with different references retain
    shared actions and average their conditional reference losses.

    Node steps are ``config.initial_step`` (the report's s/(2lambda) by
    default). The contraction guarantee requires the report's node curvature
    condition, which is not presumed for translation CE. ``max_evaluations``
    bounds actual calls to the full-batch loss function, including line searches
    in deterministic suffix solves, and raises rather than returning a partial
    policy. The pointwise loss function must not couple different batch rows.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    x, valid = tree.x.detach(), tree.valid
    p = tree.probabilities
    assert p is not None
    prefix_length = 0
    if fixed_prefix_y is not None:
        if (fixed_prefix_y.ndim != 3 or fixed_prefix_y.shape[0] != len(x)
                or fixed_prefix_y.shape[2] != x.shape[2]):
            raise ValueError("fixed_prefix_y must have shape [N,P,D]")
        prefix_length = fixed_prefix_y.shape[1]
        if prefix_length > x.shape[1] or not bool(valid[:, :prefix_length].all()):
            raise ValueError("fixed prefix cannot extend beyond a scenario path")
        if not bool(torch.isfinite(fixed_prefix_y).all()):
            raise ValueError("fixed attacked prefix must be finite")
        for t in range(prefix_length):
            for node in torch.unique(tree.node_ids[:, t]).tolist():
                values = fixed_prefix_y[tree.node_ids[:, t] == node, t]
                if not bool((values == values[0]).all()):
                    raise ValueError("fixed attacked prefix violates node sharing")

    evaluations = 0
    node_solves = 0
    node_updates = 0
    deterministic_suffix_solves = 0

    def counted_loss(y: Tensor) -> Tensor:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise NestedSolveBudgetExceeded(
                f"nested Causal Duchi exceeded max_evaluations={max_evaluations}; "
                "increase the explicit budget or use the shared-tree approximation")
        evaluations += 1
        return _losses(loss_fn, y)

    def solve_node(rows: Tensor, t: int, past: Tensor) -> tuple[Tensor, float, bool]:
        nonlocal node_solves, node_updates, deterministic_suffix_solves
        node_solves += 1
        conditional_p = torch.zeros_like(p)
        conditional_p[rows] = p[rows] / p[rows].sum()
        first = int(rows[0])
        same_length = bool((valid[rows, t:] == valid[first, t:]).all())
        deterministic = same_length and bool((x[rows, t:][valid[rows, t:]] ==
            x[first, t:][valid[first, t:]].repeat(len(rows), 1)).all())
        if deterministic:
            deterministic_suffix_solves += 1
            length = int(valid[first].sum())

            def expand_suffix(action: Tensor) -> Tensor:
                path = x.clone()
                path[rows, :t] = past[None]
                path[rows, t:length] = action[None]
                return path

            suffix, diagnostic = _shared_ascent(x[first, t:length],
                x.new_ones(length - t), expand_suffix, x, valid,
                counted_loss, config, conditional_p,
                torch.ones(length - t, dtype=torch.bool, device=x.device))
            node_updates += diagnostic["iterations"]
            return suffix, diagnostic["max_direction_residual"], diagnostic["converged"]

        current = x[first, t].clone()
        for iteration in range(config.steps + 1):
            # Recourse is recomputed for the current candidate, including after
            # the final update, and may depend only on each child's source prefix.
            continuation = x.clone()
            continuation[rows, :t] = past[None]
            continuation[rows, t] = current[None]
            child_residual = 0.0
            child_converged = True
            if t + 1 < x.shape[1]:
                surviving = rows[valid[rows, t + 1]]
                for child in torch.unique(tree.node_ids[surviving, t + 1]).tolist():
                    child_rows = surviving[tree.node_ids[surviving, t + 1] == child]
                    child_path, residual, converged = solve_node(child_rows, t + 1,
                        torch.cat((past, current[None]), dim=0).detach())
                    continuation[child_rows] = child_path[child_rows]
                    child_residual = max(child_residual, residual)
                    child_converged = child_converged and converged
            with torch.enable_grad():
                probe = current.detach().requires_grad_(True)
                candidate = continuation.detach().clone()
                candidate[rows, t] = probe[None]
                payoff = counted_loss(candidate) - config.lam * transport_cost(
                    x, candidate, valid, config.cost_scale)
                direction = torch.autograd.grad((conditional_p * payoff).sum(), probe)[0]
            if not bool(torch.isfinite(direction).all()):
                raise FloatingPointError("nested Causal Duchi direction became nonfinite")
            residual = float(direction.norm().detach())
            if residual <= config.tolerance or iteration == config.steps:
                return continuation.detach(), max(residual, child_residual), (
                    residual <= config.tolerance and child_converged)
            current = current.detach() + config.initial_step * direction.detach()
            if not bool(torch.isfinite(current).all()):
                raise FloatingPointError("nested Causal Duchi action became nonfinite")
            node_updates += 1
        raise AssertionError("unreachable nested node termination")

    y = x.clone()
    if fixed_prefix_y is not None:
        y[:, :prefix_length] = fixed_prefix_y.detach()
    max_residual = 0.0
    converged = True
    if prefix_length < x.shape[1]:
        active_rows = torch.where(valid[:, prefix_length])[0]
        for root in torch.unique(tree.node_ids[active_rows, prefix_length]).tolist():
            rows = active_rows[tree.node_ids[active_rows, prefix_length] == root]
            path, residual, root_converged = solve_node(rows, prefix_length,
                y[int(rows[0]), :prefix_length].detach())
            y[rows] = path[rows]
            max_residual = max(max_residual, residual)
            converged = converged and root_converged
    diagnostics = {
        "method": "causal_duchi_nested_conditional_ascent",
        "conditional_law": "supplied_finite_scenario_tree",
        "continuation_solution": "resolved_before_each_current_node_gradient",
        "envelope_derivative": "descendant_actions_detached",
        "deterministic_suffix_shortcut": "joint_shared_sequence_ascent",
        "node_solves": node_solves, "iterations": node_updates,
        "deterministic_suffix_solves": deterministic_suffix_solves,
        "objective_evaluations": evaluations, "max_evaluations": max_evaluations,
        "max_direction_residual": max_residual, "converged": converged,
        "exact_solve": False, "global_optimum_certified": False,
    }
    result = _result(x, valid, y, counted_loss, config, p, diagnostics)
    result.diagnostics["objective_evaluations"] = evaluations
    return result


ContinuationSampler = Callable[[Tensor], tuple[ScenarioTree, LossFunction]]


def causal_duchi_step(prefix: Tensor, previous_y: Tensor,
                      continuation_sampler: ContinuationSampler,
                      config: AttackConfig, *,
                      solver: Literal["shared", "nested", "adaptive"] = "adaptive",
                      max_evaluations: int = 10000,
                      restarts: int = 3, seed: int = 0) -> tuple[Tensor, dict[str, Any]]:
    """Commit one online action using a prefix-only conditional-law callback.

    The callback receives only the observed source prefix [t+1,D]. It must use
    a frozen training-data conditional model/bank and return scenarios and
    their own losses/references. Passing a held-out reference through a closure
    would change the information set and is forbidden by this API's contract.
    Conditional-model bias is separate from the recorded optimization residual.
    """
    if prefix.ndim != 2 or previous_y.shape != (prefix.shape[0] - 1, prefix.shape[1]):
        raise ValueError("expected source prefix [t+1,D] and attacked past [t,D]")
    tree, loss_fn = continuation_sampler(prefix.detach().clone())
    t = prefix.shape[0] - 1
    if tree.x.shape[1] <= t or not bool(tree.valid[:, :t + 1].all()):
        raise ValueError("conditional scenarios must contain the complete observed prefix")
    if not bool((tree.x[:, :t + 1] == prefix[None]).all()):
        raise ValueError("conditional scenarios must preserve the observed source prefix exactly")
    if torch.unique(tree.node_ids[:, t]).numel() != 1:
        raise ValueError("current conditional scenario action must be shared")
    fixed = previous_y[None].expand(len(tree.x), -1, -1)
    if solver == "shared":
        result = causal_duchi_attack(tree, loss_fn, config, fixed_prefix_y=fixed)
    elif solver == "nested":
        result = nested_causal_duchi_attack(tree, loss_fn, config,
            fixed_prefix_y=fixed, max_evaluations=max_evaluations)
    elif solver == "adaptive":
        result = adaptive_causal_duchi_attack(tree, loss_fn, config,
            fixed_prefix_y=fixed, max_evaluations=max_evaluations,
            restarts=restarts, seed=seed)
    else:
        raise ValueError("Causal Duchi solver must be shared, nested, or adaptive")
    return result.y[0, t].detach(), result.diagnostics


def causal_duchi_streaming_attack(x: Tensor, valid: Tensor,
                                  continuation_sampler: ContinuationSampler,
                                  loss_fn: LossFunction, config: AttackConfig, *,
                                  solver: Literal["shared", "nested", "adaptive"] = "adaptive",
                                  max_evaluations: int = 10000,
                                  restarts: int = 3, seed: int = 0) -> AttackResult:
    """Online deployment; actual test losses are evaluated only after actions."""
    _validate_paths(x, valid)
    y = x.detach().clone()
    node_diagnostics = []
    for i in range(len(x)):
        for t in range(x.shape[1]):
            if not bool(valid[i, t]):
                break
            y[i, t], diagnostic = causal_duchi_step(x[i, :t + 1], y[i, :t],
                continuation_sampler, config, solver=solver, max_evaluations=max_evaluations,
                restarts=restarts, seed=seed)
            node_diagnostics.append(diagnostic)
    diagnostics = {
        "method": ("adaptive_causal_duchi_prefix_conditional_deployment" if solver == "adaptive"
                   else "legacy_causal_duchi_prefix_conditional_deployment"),
        "solver": solver,
        "steps_committed": len(node_diagnostics),
        "max_direction_residual": max(d["max_direction_residual"] for d in node_diagnostics),
        "converged": all(d["converged"] for d in node_diagnostics),
        "exact_solve": False, "global_optimum_certified": False,
        "node_diagnostics": node_diagnostics,
    }
    return _result(x, valid, y, loss_fn, config, _probabilities(x, None), diagnostics)


class PositiveLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.full((output_dim, input_dim), -3.0))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, F.softplus(self.raw_weight))


class StronglyConvexPICNN(nn.Module):
    """Convex in action, unconstrained in frozen context, identity initialized.

    K(y;h) = (lambda/s)[.5 sum_j q_j y_j² + r(y;h) + a(h)^T y],
    q_j >= curvature_floor > 0, with convex softplus PICNN r. Unlike a fixed
    lambda||y||² plus convex residual, the learnable q can represent Bellman
    energies whose curvature is below 2lambda/s. At initialization q=2 and
    both residual action derivatives vanish, giving grad K*(2lambda/s x)=x.
    """

    def __init__(self, action_dim: int, context_dim: int, hidden_dim: int = 32,
                 depth: int = 2, curvature_floor: float = 0.1) -> None:
        super().__init__()
        if min(action_dim, context_dim, hidden_dim, depth) < 1:
            raise ValueError("PICNN dimensions and depth must be positive")
        if not 0 < curvature_floor < 2:
            raise ValueError("curvature_floor must lie in (0,2) for identity initialization")
        self.curvature_floor = float(curvature_floor)
        # Zero raw curvature gives exactly q=2 even after .double()/.float().
        # Initializing an inverse-softplus constant in float32 and later
        # promoting it would otherwise introduce a spurious nonidentity map.
        self.raw_curvature = nn.Parameter(torch.zeros(action_dim))
        self.action_layers = nn.ModuleList(nn.Linear(action_dim, hidden_dim, bias=False) for _ in range(depth))
        self.context_layers = nn.ModuleList(nn.Linear(context_dim, hidden_dim) for _ in range(depth))
        self.action_gates = nn.ModuleList(nn.Linear(context_dim, action_dim) for _ in range(depth))
        self.positive_layers = nn.ModuleList(PositiveLinear(hidden_dim, hidden_dim) for _ in range(depth - 1))
        self.positive_gates = nn.ModuleList(nn.Linear(context_dim, hidden_dim) for _ in range(depth - 1))
        self.output_positive = PositiveLinear(hidden_dim, 1)
        self.affine = nn.Linear(context_dim, action_dim)
        for layer in self.action_layers:
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, action: Tensor, context: Tensor, lam: float = 1.0,
                cost_scale: float = 1.0) -> Tensor:
        if action.shape[:-1] != context.shape[:-1]:
            raise ValueError("action and context batch shapes must agree")
        if lam <= 0 or cost_scale <= 0:
            raise ValueError("lambda and cost scale must be positive")
        z = None
        for j, (a, h, gate) in enumerate(zip(self.action_layers, self.context_layers, self.action_gates)):
            pre = a(action * torch.tanh(gate(context))) + h(context)
            if z is not None:
                pre = pre + self.positive_layers[j - 1](z * F.softplus(self.positive_gates[j - 1](context)))
            z = F.softplus(pre)
        assert z is not None
        q = self.curvature_floor + (2.0 - self.curvature_floor) * (F.softplus(self.raw_curvature) / math.log(2.0))
        return (lam / cost_scale) * (0.5 * (q * action.square()).sum(-1) +
            self.output_positive(z).squeeze(-1) + (self.affine(context) * action).sum(-1))


def conjugate_solve(energy: StronglyConvexPICNN, context: Tensor, source: Tensor,
                    *, lam: float, cost_scale: float = 1.0, steps: int = 20,
                    tolerance: float = 1e-6, differentiable: bool = False,
                    backtracking_steps: int = 20) -> tuple[Tensor, dict[str, Any]]:
    """Finite solve of grad_y K(y;h)=2lambda/s x, with rowwise line search.

    Differentiation unrolls the actual finite solver; it does not pretend to
    provide exact implicit derivatives of an unconverged conjugate root. The
    line-search choices are detached. Every differentiable iteration runs even
    at the identity root so the initialized attack can receive gradients.
    Rowwise step selection and convergence prevent batch-composition leakage.
    """
    if steps < 1 or tolerance <= 0 or backtracking_steps < 1 or lam <= 0 or cost_scale <= 0:
        raise ValueError("invalid conjugate solver configuration")
    # An independent graph node is essential: differentiating with respect to
    # ``source`` itself would also differentiate dual=2lambda/s*source, rather
    # than freezing the conjugate variable in the node stationarity equation.
    action = source.clone() if differentiable else source.detach().clone()
    dual = 2.0 * lam / cost_scale * source
    active = torch.ones(source.shape[:-1], dtype=torch.bool, device=source.device)
    failures = torch.zeros_like(active)
    completed = 0
    def objective(point):
        if hasattr(energy, 'conjugate_objective'):
            return energy.conjugate_objective(point, context, source, lam, cost_scale)
        return energy(point, context, lam, cost_scale) - (dual * point).sum(-1)
    with torch.enable_grad():
        for _ in range(steps):
            if not action.requires_grad:
                action = action.detach().requires_grad_(True)
            value = objective(action)
            gradient = torch.autograd.grad(value.sum(), action, create_graph=differentiable)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("PICNN conjugate gradient became nonfinite")
            if not differentiable:
                active = active & (gradient.detach().norm(dim=-1) > tolerance)
                if not bool(active.any()):
                    break
            step = source.new_full(source.shape[:-1], cost_scale / (2.0 * lam))
            accepted = ~active
            for _ in range(backtracking_steps):
                with torch.no_grad():
                    proposal = action.detach() - step[..., None] * gradient.detach()
                    proposal_value = objective(proposal)
                    slack = 8 * torch.finfo(source.dtype).eps * value.detach().abs().clamp_min(1)
                    good = torch.isfinite(proposal_value) & (proposal_value <=
                        value.detach() - 1e-4 * step * gradient.detach().square().sum(-1) + slack)
                accepted = accepted | good
                if bool(accepted.all()):
                    break
                step = torch.where(accepted, step, step * 0.5)
            failures = failures | (active & ~accepted)
            effective_step = step * (active & accepted)
            action = action - effective_step[..., None] * gradient
            completed += 1
            if not differentiable:
                action = action.detach()
        probe = action if action.requires_grad else action.detach().requires_grad_(True)
        value = objective(probe)
        residual_vector = torch.autograd.grad(value.sum(), probe, retain_graph=differentiable)[0]
    residuals = residual_vector.detach().norm(dim=-1)
    diagnostics = {
        "iterations": completed, "max_conjugate_residual": float(residuals.max()),
        "converged": bool((residuals <= tolerance).all()),
        "line_search_failed": bool(failures.any()), "exact_solve": False,
        "derivative": "finite_unrolled_solver" if differentiable else "none",
        "strong_convexity_lower_bound": lam / cost_scale * energy.curvature_floor,
    }
    return (action if differentiable else action.detach()), diagnostics


class PICNNAttacker(nn.Module):
    """Shared streaming action energy with explicit causal/full-source context.

    The source GRU receives one source chunk at a time; the attacked-history
    GRU receives only previously committed actions. No target reference,
    utterance length, batch statistics, or future padding enters causal context.
    Both versions retain the source-prefix summary. The anticipative comparison
    additionally supplies a full-source summary in a context slot masked to
    zero for the causal attack. Thus its function class contains the causal
    class (set the extra input weights to zero), with identical parameter counts.
    """

    def __init__(self, action_dim: int, context_dim: int = 32, hidden_dim: int = 32,
                 depth: int = 2, threat: Literal["causal", "anticipative"] = "causal",
                 solver_steps: int = 20, solver_tolerance: float = 1e-6,
                 architecture: str = 'legacy') -> None:
        super().__init__()
        if threat not in ("causal", "anticipative"):
            raise ValueError("threat must be causal or anticipative")
        self.threat = threat
        self.context_dim = context_dim
        self.solver_steps = solver_steps
        self.solver_tolerance = solver_tolerance
        self.source_encoder = nn.GRUCell(action_dim, context_dim)
        self.history_encoder = nn.GRUCell(action_dim, context_dim)
        self.context_network = nn.Sequential(nn.Linear(3 * context_dim, context_dim), nn.Tanh())
        self.energy = StronglyConvexPICNN(action_dim, context_dim, hidden_dim, depth)
        self.input_normalization = nn.Identity()
        if architecture == 'lastquad':
            from .lastquad import LastQuadPICNN, initialize_recurrent, lecun_
            self.energy = LastQuadPICNN(action_dim, context_dim, hidden_dim, depth)
            # Per-observed-block normalization only, with no cross-time or
            # cross-example statistics. Transport still uses original states.
            self.input_normalization = nn.LayerNorm(action_dim, elementwise_affine=False)
            initialize_recurrent(self.source_encoder)
            initialize_recurrent(self.history_encoder)
            lecun_(self.context_network[0])
        elif architecture != 'legacy':
            raise ValueError('Unknown PICNN architecture')
        self.architecture = architecture

    def forward(self, x: Tensor, valid: Tensor, lam: float, cost_scale: float = 1.0,
                differentiable: bool = False) -> tuple[Tensor, dict[str, Any]]:
        _validate_paths(x, valid)
        if not differentiable:
            x = x.detach()
        source_state = x.new_zeros((len(x), self.context_dim))
        source_contexts = []
        for t in range(x.shape[1]):
            next_state = self.source_encoder(self.input_normalization(x[:, t]), source_state)
            source_state = torch.where(valid[:, t, None], next_state, source_state)
            source_contexts.append(source_state)
        history = torch.zeros_like(source_state)
        attacked, node_diagnostics = [], []
        for t in range(x.shape[1]):
            future_summary = torch.zeros_like(source_state) if self.threat == "causal" else source_contexts[-1]
            context = self.context_network(torch.cat((source_contexts[t], future_summary, history), dim=-1))
            rows = torch.where(valid[:, t])[0]
            if rows.numel():
                current, diagnostic = conjugate_solve(self.energy, context[rows], x[rows, t],
                    lam=lam, cost_scale=cost_scale, steps=self.solver_steps,
                    tolerance=self.solver_tolerance, differentiable=differentiable)
                action = x[:, t].clone()
                action[rows] = current
                node_diagnostics.append(diagnostic)
            else:
                action = x[:, t]
            attacked.append(action)
            next_history = self.history_encoder(self.input_normalization(action), history)
            history = torch.where(valid[:, t, None], next_history, history)
        y = torch.stack(attacked, dim=1)
        diagnostics = {
            "method": self.threat + "_picnn", "exact_solve": False,
            "max_conjugate_residual": max(d["max_conjugate_residual"] for d in node_diagnostics),
            "converged": all(d["converged"] for d in node_diagnostics),
            "line_search_failed": any(d["line_search_failed"] for d in node_diagnostics),
            "solver_steps": self.solver_steps, "solver_tolerance": self.solver_tolerance,
            "derivative": "finite_unrolled_solver" if differentiable else "none",
            "strong_convexity_lower_bound": lam / cost_scale * self.energy.curvature_floor,
            "global_optimum_certified": False,
        }
        return (y if differentiable else y.detach()), diagnostics

    def attack(self, x: Tensor, valid: Tensor, loss_fn: LossFunction,
               config: AttackConfig, differentiable: bool = False,
               probabilities: Tensor | None = None) -> AttackResult:
        y, diagnostics = self(x, valid, config.lam, config.cost_scale, differentiable)
        return _result(x, valid, y, loss_fn, config, _probabilities(x, probabilities),
            diagnostics, differentiable=differentiable)


class RecurrentAttacker(nn.Module):
    """One-pass triangular source map from the recurrent-adversary report.

    The recurrent cell receives the current nominal block and a stage index
    divided by a fixed public scale. The eventual source length, references,
    previous decoder outputs, and batch statistics are never network inputs.
    A zero displacement projection gives the exact identity map initially.
    """

    threat = "causal_rnn"
    information = "causal"

    def __init__(self, action_dim: int, hidden_dim: int = 64,
                 head_dim: int = 64, time_scale: float = 128.0) -> None:
        super().__init__()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (action_dim, hidden_dim, head_dim)):
            raise ValueError("RNN dimensions must be positive integers")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("RNN time_scale must be positive and finite")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.time_scale = float(time_scale)
        self.input_normalization = nn.LayerNorm(action_dim, elementwise_affine=False)
        self.source_encoder = nn.GRUCell(action_dim + 1, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, head_dim), nn.Tanh(),
            nn.Linear(head_dim, action_dim))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: Tensor, valid: Tensor, lam: float,
                cost_scale: float = 1.0, differentiable: bool = False
                ) -> tuple[Tensor, dict[str, Any]]:
        _validate_paths(x, valid)
        if not math.isfinite(lam) or lam <= 0 or not math.isfinite(cost_scale) or cost_scale <= 0:
            raise ValueError("RNN penalty parameters must be positive and finite")
        x = x.detach()
        with torch.set_grad_enabled(differentiable):
            state = x.new_zeros((len(x), self.hidden_dim))
            actions = []
            for t in range(x.shape[1]):
                # This coordinate is unchanged when a future source block is
                # appended; normalizing by this example's T would leak length.
                stage = x.new_full((len(x), 1), (t + 1) / self.time_scale)
                observed = torch.cat((self.input_normalization(x[:, t]), stage), dim=-1)
                next_state = self.source_encoder(observed, state)
                state = torch.where(valid[:, t, None], next_state, state)
                shift = self.head(torch.cat((state, stage), dim=-1))
                actions.append(torch.where(valid[:, t, None], x[:, t] + shift, x[:, t]))
            y = torch.stack(actions, dim=1)
        if not bool(torch.isfinite(y).all()):
            raise FloatingPointError("RNN attack actions became nonfinite")
        diagnostics = {
            "method": "causal_rnn",
            "information": "source_prefix_only",
            "time_coordinate": "one_based_stage_over_fixed_scale",
            "time_scale": self.time_scale,
            "one_pass": True,
            "global_optimum_certified": False,
        }
        return (y if differentiable else y.detach()), diagnostics

    def attack(self, x: Tensor, valid: Tensor, loss_fn: LossFunction,
               config: AttackConfig, differentiable: bool = False,
               probabilities: Tensor | None = None) -> AttackResult:
        y, diagnostics = self(x, valid, config.lam, config.cost_scale, differentiable)
        return _result(x, valid, y, loss_fn, config, _probabilities(x, probabilities),
            diagnostics, differentiable=differentiable)


def train_recurrent_step(attacker: RecurrentAttacker, optimizer: torch.optim.Optimizer,
                         x: Tensor, valid: Tensor, loss_fn: LossFunction,
                         config: AttackConfig, probabilities: Tensor | None = None,
                         gradient_scale: float = 1.0) -> dict[str, Any]:
    """Ascend training CE minus additive transport cost through the GRU pass."""
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be positive and finite")
    optimizer.zero_grad(set_to_none=True)
    result = attacker.attack(x.detach(), valid, loss_fn, config,
        differentiable=True, probabilities=probabilities)
    p = _probabilities(x, probabilities)
    objective = (p * (result.losses - config.lam * result.costs)).sum()
    parameters = [parameter for parameter in attacker.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(-objective, parameters, allow_unused=True)
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("RNN training gradient became nonfinite")
        parameter.grad = None if gradient is None else gradient * gradient_scale
    optimizer.step()
    result.diagnostics["training_gradient_norm"] = float(torch.sqrt(sum(
        (gradient.detach().square().sum() for gradient in gradients if gradient is not None),
        x.new_zeros(()))))
    return result.diagnostics


def train_picnn_step(attacker: PICNNAttacker, optimizer: torch.optim.Optimizer,
                     x: Tensor, valid: Tensor, loss_fn: LossFunction,
                     config: AttackConfig, probabilities: Tensor | None = None,
                     gradient_scale: float = 1.0) -> dict[str, Any]:
    """One adversary ascent step; caller freezes defender parameters/eval mode.

    Only attacker parameter gradients are requested, so defender ``.grad``
    buffers are not populated even if the caller leaves requires_grad enabled.
    The returned diagnostics describe the pre-update finite attack solve.
    """
    optimizer.zero_grad(set_to_none=True)
    result = attacker.attack(x.detach(), valid, loss_fn, config,
        differentiable=True, probabilities=probabilities)
    p = _probabilities(x, probabilities)
    objective = (p * (result.losses - config.lam * result.costs)).sum()
    parameters = [parameter for parameter in attacker.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(-objective, parameters, allow_unused=True)
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError('gradient_scale must be positive and finite')
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("PICNN training gradient became nonfinite")
        parameter.grad = None if gradient is None else gradient * gradient_scale
    optimizer.step()
    return result.diagnostics
