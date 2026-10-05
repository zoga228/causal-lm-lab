"""Behavioral invariants: future independence, backward pass, and safe serialization."""
import tempfile
import unittest
import torch
from model import ByteLM, Config


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = ByteLM(Config(width=32, layers=2, heads=4, context=16)).eval()

    def test_future_tokens_cannot_change_prefix_logits(self):
        x = torch.randint(0, 256, (2, 16))
        modified = x.clone()
        modified[:, 8:] = (modified[:, 8:] + 70) % 256
        torch.testing.assert_close(self.model(x)[:, :8], self.model(modified)[:, :8], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(self.model(x)[:, 8:], self.model(modified)[:, 8:]))

    def test_no_future_gradient_path(self):
        x = torch.randint(0, 256, (1, 16))
        captured = []
        hook = self.model.embedding.register_forward_hook(lambda m, i, out: (out.retain_grad(), captured.append(out)) and None)
        self.model(x)[0, 5, 10].backward()
        self.assertEqual(0, torch.count_nonzero(captured[0].grad[:, 6:]).item())
        self.assertGreater(torch.count_nonzero(captured[0].grad[:, :6]).item(), 0)
        hook.remove()

    def test_safe_checkpoint_round_trip(self):
        x = torch.randint(0, 256, (1, 8))
        with tempfile.TemporaryDirectory() as d:
            self.model.save_pretrained(d)
            restored = ByteLM.from_pretrained(d)
            torch.testing.assert_close(self.model(x), restored(x), atol=0, rtol=0)

    def test_loss_backward_is_finite(self):
        x = torch.randint(0, 256, (2, 16))
        loss = torch.nn.functional.cross_entropy(self.model(x).reshape(-1, 256), x.reshape(-1))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
