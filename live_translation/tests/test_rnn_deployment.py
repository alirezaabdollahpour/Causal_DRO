"""The saved recurrent policy must act from committed source prefixes."""

import tempfile
import unittest
from pathlib import Path

import torch

from live_translation.attacks import RecurrentAttacker
from live_translation.rnn_deployment import load_causal_rnn


class RNNDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "causal_rnn.pt"
        torch.manual_seed(14)
        self.attacker = RecurrentAttacker(3, hidden_dim=5, head_dim=4,
                                          time_scale=40)
        with torch.no_grad():
            self.attacker.head[-1].weight.fill_(0.03)
            self.attacker.head[-1].bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
        self.payload = dict(
            state_dict=self.attacker.state_dict(), scale=0.7,
            config=dict(wait_k=[2], lam=3.0, cost_scale=2.0,
                        rnn_hidden_dim=5, rnn_head_dim=4, rnn_time_scale=40),
            wait_k=2, threat="causal_rnn")
        torch.save(self.payload, self.path)

    def test_loader_reproduces_calibrated_prefix_policy(self):
        policy = load_causal_rnn(self.path, 3, "cpu", 2)
        self.assertEqual(policy.information, "causal")
        self.assertFalse(any(parameter.requires_grad for parameter in policy.parameters()))
        x = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        raw, _ = self.attacker(x, valid, 3.0, 2.0)
        with torch.no_grad():
            y = policy(x, valid)
            first = policy(x[:, :1], valid[:, :1])
            second = policy(x[:, :2], valid[:, :2])
        torch.testing.assert_close(y, x + 0.7 * (raw - x), rtol=0, atol=1e-6)
        torch.testing.assert_close(first, y[:, :1], rtol=0, atol=1e-6)
        torch.testing.assert_close(second, y[:, :2], rtol=0, atol=1e-6)

    def test_padding_does_not_change_committed_actions(self):
        policy = load_causal_rnn(self.path, 3, "cpu", 2)
        x = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [9., 9., 9.]]])
        valid = torch.tensor([[True, True, False]])
        y = policy(x, valid)
        torch.testing.assert_close(y[:, :2], policy(x[:, :2], valid[:, :2]),
                                   rtol=0, atol=1e-6)
        torch.testing.assert_close(y[:, 2], x[:, 2], rtol=0, atol=0)

    def test_wrong_wait_k_or_uncalibrated_artifact_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wait-k"):
            load_causal_rnn(self.path, 3, "cpu", 1)
        for change in (dict(threat="anticipative_rnn"), dict(scale=float("nan")),
                       dict(scale=-1.0)):
            bad = dict(self.payload, **change)
            torch.save(bad, self.path)
            with self.assertRaises(ValueError):
                load_causal_rnn(self.path, 3, "cpu", 2)

    def test_wrong_source_dimension_is_rejected(self):
        with self.assertRaises(RuntimeError):
            load_causal_rnn(self.path, 4, "cpu", 2)


if __name__ == "__main__":
    unittest.main()
