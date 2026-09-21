"""Coverage checks for the full-corpus training schedule."""
import json
import sys

import pytest

import live_translation.benchmark as benchmark_module
from live_translation.benchmark import Benchmark, BenchmarkConfig


class Rows:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)


def make_benchmark(*, explicit_steps=None):
    benchmark = object.__new__(Benchmark)
    benchmark.c = BenchmarkConfig(batch_size=3, training_singleton_seconds=20,
                                  warmup_steps=explicit_steps)
    benchmark.datasets = {"train": Rows([
        {"id": f"train:{index}", "duration": 25 if index == 4 else 5}
        for index in range(7)])}
    benchmark._sample_permutations = {}
    benchmark._training_batch_count = None
    benchmark.batch = lambda split, indices: list(indices)
    return benchmark


@pytest.mark.parametrize("explicit_steps", [None, 3])
def test_one_epoch_sees_every_training_row_once(explicit_steps):
    benchmark = make_benchmark(explicit_steps=explicit_steps)
    steps = benchmark.total_steps("warmup")
    assert steps == 3
    batches = [benchmark.sample(step, stream=10, phase="warmup")
               for step in range(steps)]
    assert sorted(index for batch in batches for index in batch) == list(range(7))
    assert any(batch == [4] for batch in batches)


def test_full_corpus_rejects_steps_shorter_than_one_epoch():
    benchmark = make_benchmark(explicit_steps=2)
    with pytest.raises(ValueError, match="less than one complete training epoch"):
        benchmark.total_steps("warmup")


def test_cli_epoch_override_clears_configured_step_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"warmup_steps": 3}))
    captured = {}

    class Capture:
        def __init__(self, config):
            captured["config"] = config

        def run(self, stage):
            captured["stage"] = stage

    monkeypatch.setattr(benchmark_module, "Benchmark", Capture)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--config", str(config_path),
                                      "--stage", "train", "--warmup-epochs", "2"])
    benchmark_module.main()
    assert captured["config"].warmup_epochs == 2
    assert captured["config"].warmup_steps is None
    assert captured["stage"] == "train"


def test_cli_rejects_both_step_and_epoch_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--warmup-steps", "3",
                                      "--warmup-epochs", "2"])
    with pytest.raises(SystemExit, match="2"):
        benchmark_module.main()
