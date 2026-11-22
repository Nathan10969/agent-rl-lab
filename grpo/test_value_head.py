"""价值头和动作前 Qwen 隐藏表示：CPU、随机迷你模型，不加载真实模型。"""
import importlib.util
import io
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM


def implementation(test):
    test.assertIsNotNone(importlib.util.find_spec("value_head"), "Qwen 价值头尚未实现")
    import value_head
    return value_head


class ValueHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
        self.model = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        )).eval()
        self.turn = {"prompt_ids": [0, 0, 4, 5, 6], "prompt_mask": [0, 0, 1, 1, 1],
                     "token_ids": [7, 8]}

    def test_left_padding_and_future_tokens_do_not_change_prefix_features(self):
        m = implementation(self)
        padded = m.extract_prompt_hidden(self.model, self.turn)
        compact = m.extract_prompt_hidden(self.model, {"prompt_ids": [4, 5, 6], "prompt_mask": [1, 1, 1]})
        different_future = m.extract_prompt_hidden(self.model, {**self.turn, "token_ids": [9, 10, 11]})
        self.assertEqual(tuple(padded.shape), (1, 16))
        self.assertFalse(padded.requires_grad)
        self.assertTrue(torch.allclose(padded, compact, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(padded, different_future))

    def test_reject_invalid_masks_and_training_mode(self):
        m = implementation(self)
        invalid = [([], []), ([1, 2], [1]), ([1, 2], [1, 0]), ([1, 2], [0, 0]),
                   ([1, 2, 3], [1, 0, 1]), ([1, 2], [2, 1])]
        for ids, mask in invalid:
            with self.subTest(ids=ids, mask=mask), self.assertRaises(ValueError):
                m.extract_prompt_hidden(self.model, {"prompt_ids": ids, "prompt_mask": mask})
        self.model.train()
        with self.assertRaises(ValueError):
            m.extract_prompt_hidden(self.model, self.turn)

    def test_value_head_update_is_isolated_and_reload_matches(self):
        m = implementation(self)
        head = m.ValueHead(16)
        turns = [self.turn, {"prompt_ids": [4, 5, 7], "prompt_mask": [1, 1, 1]}]
        before_model = {name: param.detach().clone() for name, param in self.model.named_parameters()}
        before_head = {name: param.detach().clone() for name, param in head.named_parameters()}
        values = m.predict_trajectory_values(self.model, head, turns)
        self.assertEqual(tuple(values.shape), (2,))
        self.assertTrue(torch.isfinite(values).all())
        optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
        optimizer.zero_grad()
        ((values - torch.tensor([0.71, 0.9])) ** 2).mean().backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(before_head[n], p) for n, p in head.named_parameters()))
        self.assertTrue(all(torch.equal(before_model[n], p) for n, p in self.model.named_parameters()))

        buffer = io.BytesIO()
        torch.save(head.state_dict(), buffer)
        buffer.seek(0)
        restored = m.ValueHead(16)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(m.predict_trajectory_values(self.model, head, turns),
                                    m.predict_trajectory_values(self.model, restored, turns)))
        with self.assertRaises(ValueError):
            m.predict_trajectory_values(self.model, head, [])

    def test_step_advantages_are_aligned_and_detached(self):
        m = implementation(self)
        self.assertTrue(hasattr(m, "step_advantages"), "逐步优势计算尚未实现")
        values = torch.tensor([0.8, 0.75, 0.8], requires_grad=True)
        returns = torch.tensor([0.71, 0.9, 1.0])
        advantages = m.step_advantages(returns, values)
        self.assertTrue(torch.allclose(advantages, torch.tensor([-0.09, 0.15, 0.2]), atol=1e-6))
        self.assertFalse(advantages.requires_grad)
        with self.assertRaises(ValueError):
            m.step_advantages(returns[:2], values)
        with self.assertRaises(ValueError):
            m.step_advantages(torch.tensor([float("nan")]), values[:1])


if __name__ == "__main__":
    unittest.main()
