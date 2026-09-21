"""Run the small synthetic example or the MuST-C English–German pilot."""

import argparse
from dataclasses import replace
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument("--output", help="Directory for results; must not contain a completed run")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--lambda", dest="lam", type=float,
                        help="Penalty weight on transport cost")
    parser.add_argument("--warmup-steps", type=int,
                        help="Clean defender updates before adversarial training")
    parser.add_argument("--defender-steps", type=int,
                        help="Defender updates after warmup")
    parser.add_argument("--attack-inner-steps", type=int,
                        help="Attacker updates per defender update")
    parser.add_argument("--attack-fit-steps", type=int,
                        help="Attacker updates when fitting each evaluation condition")
    parser.add_argument("--attack-steps", type=int,
                        help="Optimization steps for the direct anticipative attack")
    parser.add_argument("--picnn-solver-steps", type=int,
                        help="PICNN forward solver steps")
    parser.add_argument("--check", action="store_true",
                        help="Validate configuration and required input paths, then exit")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        overrides = {name: getattr(args, name) for name in (
            "output", "seed", "lam", "warmup_steps", "defender_steps",
            "attack_inner_steps", "attack_fit_steps", "attack_steps",
            "picnn_solver_steps") if getattr(args, name) is not None}
        config = replace(config, **overrides)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.check:
        print("Configuration and input paths are valid")
        return
    from .runner import Experiment
    if config.backend == "toy" and config.device == "cpu":
        import torch
        torch.set_num_threads(1)
    result = Experiment(config).run()
    from .render import render
    render(config.output)
    print(json.dumps(dict(status=result["status"], output=config.output,
                          conditions=len(result["rows"]))))


if __name__ == "__main__":
    main()
