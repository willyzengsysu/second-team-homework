"""贝塔估计的基本性质测试（不依赖网络数据）。

核心命题（任务书要求解释的那一条）：
    含截距 OLS 的斜率 与 定义法 cov/var 在同样本下恒等。

这里用合成数据验证，并检查 HAC 标准误与分组切分是否正确。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import allocate  # noqa: E402
from src.s4_beta import beta_by_definition, ols_beta  # noqa: E402


def make_data(n: int = 60, beta: float = 1.3, alpha: float = 0.002, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(0.005, 0.06, n))
    y = pd.Series(alpha + beta * x + rng.normal(0, 0.04, n))
    return y, x


class TestBetaIdentity(unittest.TestCase):
    def test_ols_slope_equals_cov_over_var(self):
        """同一样本、同一 n−1 分母下，两种算法之差应仅为浮点误差。"""
        for seed in range(5):
            y, x = make_data(seed=seed, beta=0.6 + 0.3 * seed)
            ols = ols_beta(y, x)
            d = beta_by_definition(y, x)
            self.assertIsNotNone(ols)
            self.assertAlmostEqual(ols["beta"], d["beta_def"], places=10)

    def test_recovers_true_beta_and_alpha(self):
        y, x = make_data(n=5000, beta=1.25, alpha=0.003, seed=42)
        ols = ols_beta(y, x)
        self.assertAlmostEqual(ols["beta"], 1.25, places=1)
        self.assertAlmostEqual(ols["alpha"], 0.003, places=2)

    def test_matches_statsmodels_directly(self):
        y, x = make_data(n=120, beta=0.9, seed=7)
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        ols = ols_beta(y, x)
        self.assertAlmostEqual(ols["beta"], float(fit.params.iloc[1]), places=12)
        self.assertAlmostEqual(ols["alpha"], float(fit.params.iloc[0]), places=12)

    def test_hac_only_changes_standard_error(self):
        """HAC 只影响标准误，不影响点估计。"""
        y, x = make_data(n=120, beta=1.1, seed=3)
        plain = ols_beta(y, x)
        hac = ols_beta(y, x, hac_lags=6)
        self.assertAlmostEqual(plain["beta"], hac["beta"], places=12)
        self.assertNotEqual(plain["se_beta"], hac["se_beta"])

    def test_missing_values_paired(self):
        """缺失值必须成对剔除，否则定义法与 OLS 会不一致。"""
        y, x = make_data(n=80, seed=11)
        y.iloc[5] = np.nan
        x.iloc[9] = np.nan
        ols = ols_beta(y, x)
        d = beta_by_definition(y, x)
        self.assertEqual(ols["n_obs"], 78)
        self.assertEqual(ols["n_obs"], d["n_obs"])
        self.assertAlmostEqual(ols["beta"], d["beta_def"], places=10)


class TestAllocation(unittest.TestCase):
    def test_quota_sums_and_caps(self):
        counts = pd.Series({"A": 100, "B": 50, "C": 3, "D": 1})
        alloc = allocate(counts, 40)
        self.assertEqual(int(alloc.sum()), 40)
        self.assertTrue((alloc <= counts).all(), "配额不得超过层内股票数")

    def test_never_exceeds_population(self):
        counts = pd.Series({"A": 2, "B": 2})
        alloc = allocate(counts, 100)
        self.assertEqual(int(alloc.sum()), 4)


if __name__ == "__main__":
    unittest.main()
