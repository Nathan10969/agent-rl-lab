"""核对价值头示例中的手算与 PyTorch 梯度。"""
import importlib.util
import unittest


class ValueHeadDemoTests(unittest.TestCase):
    def test_one_sgd_step_matches_hand_calculation(self):
        self.assertIsNotNone(importlib.util.find_spec("value_head_demo"), "价值头演示尚未实现")
        from value_head_demo import run_demo

        result = run_demo()
        self.assertAlmostEqual(result["prediction_before"], 0.5)
        self.assertAlmostEqual(result["loss_before"], 0.10205)
        for name, expected in {"w1": -1.22, "w2": -0.61, "b": -0.61}.items():
            self.assertAlmostEqual(result["gradients"][name], expected)
        for name, expected in {"w1": 0.261, "w2": -0.0695, "b": 0.2305}.items():
            self.assertAlmostEqual(result["parameters_after"][name], expected)
        self.assertAlmostEqual(result["prediction_after"], 0.683)
        self.assertAlmostEqual(result["loss_after"], 0.023909)
        self.assertLess(result["loss_after"], result["loss_before"])


if __name__ == "__main__":
    unittest.main()
