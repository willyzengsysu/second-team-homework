"""回归测试：样本币种、历史信息边界、未知缺月和退出月份。"""
import unittest
import numpy as np
import pandas as pd

from src.s1_universe import a_share_mask, historical_industry
from src.s3_clean import monthly_observations
from src.s5_portfolio import ols_alpha_beta


class TestUniverseRules(unittest.TestCase):
    def test_b_shares_never_enter_a_share_universe(self):
        codes = pd.Series(["000001", "600000", "688001", "300001", "200413", "900951"])
        self.assertEqual(a_share_mask(codes).tolist(), [True, True, True, True, False, False])

    def test_industry_cannot_come_from_future(self):
        d = pd.DataFrame({"code": ["sh.600000"], "industry": ["J66货币金融服务"], "updateDate": ["2020-01-01"]})
        with self.assertRaises(ValueError):
            historical_industry(d, "2019-12-31")
        d["updateDate"] = "2019-12-30"
        self.assertEqual(historical_industry(d, "2019-12-31").code.iloc[0], "600000")

    def test_undated_industry_rejected(self):
        with self.assertRaises(ValueError):
            historical_industry(pd.DataFrame({"code": ["600000"], "industry": ["J66"]}), "2019-12-31")


class TestMissingRules(unittest.TestCase):
    def setUp(self):
        self.grid = pd.period_range("2020-01", "2020-05", freq="M")
        self.price = pd.DataFrame({"date": ["2020-01-31", "2020-03-31", "2020-04-30"], "close": [100., 110., 121.]})

    def test_unknown_gap_not_zero_and_resumption_not_one_month_return(self):
        for zero in [False, True]:
            d = monthly_observations(self.price, self.grid, zero_marked=zero)
            self.assertTrue(np.isnan(d.ret.iloc[1]))
            self.assertTrue(np.isnan(d.ret.iloc[2]))
            self.assertAlmostEqual(d.ret.iloc[3], .1)
            self.assertFalse(d.suspended.any())

    def test_tail_is_unknown_until_delist_not_an_automatic_exit(self):
        d = monthly_observations(self.price.iloc[:1], self.grid, delist_date="2020-04-15")
        self.assertEqual(d.return_status.iloc[2], "unknown_tail")
        self.assertEqual(d.return_status.iloc[3], "unknown_tail")
        self.assertEqual(d.return_status.iloc[4], "post_delist")

    def test_ipo_does_not_become_a_suspension(self):
        d = monthly_observations(self.price.iloc[1:], self.grid, list_date="2020-03-17")
        self.assertEqual(d.return_status.iloc[0], "pre_listing")
        self.assertTrue(np.isnan(d.ret.iloc[2]))
        self.assertFalse(d.suspended.any())

    def test_only_verified_suspension_can_carry_price(self):
        e = pd.DataFrame([{"period": "2020-02", "event": "suspension", "value": np.nan, "source": "独立停牌公告"}])
        d = monthly_observations(self.price, self.grid, events=e, zero_marked=True)
        self.assertEqual(d.ret.iloc[1], 0)
        self.assertTrue(d.suspended.iloc[1])
        self.assertAlmostEqual(d.ret.iloc[2], .1)
        strict = monthly_observations(self.price, self.grid, events=e)
        self.assertTrue(np.isnan(strict.ret.iloc[1]))

    def test_price_threshold_is_effective(self):
        q = pd.DataFrame({"date": ["2020-01-31", "2020-02-28"], "close": [100., .001]})
        d = monthly_observations(q, self.grid)
        self.assertEqual(d.return_status.iloc[1], "invalid_price")
        self.assertTrue(np.isnan(d.ret.iloc[1]))

    def test_zero_volume_month_is_not_a_verified_suspension_or_traded_quote(self):
        q = pd.DataFrame({"date": ["2020-01-31", "2020-02-28", "2020-03-31"],
                          "close": [100., 100., 110.], "volume": [100, 0, 120]})
        d = monthly_observations(q, self.grid)
        self.assertEqual(d.return_status.iloc[1], "source_zero_volume")
        self.assertFalse(d.quote_observed.iloc[1])
        self.assertFalse(d.suspended.any())
        self.assertTrue(np.isnan(d.ret.iloc[2]))

    def test_terminal_return_requires_valid_value(self):
        e = pd.DataFrame([{"period": "2020-05", "event": "terminal_return", "value": -1., "source": "终止收益核查"}])
        d = monthly_observations(self.price, self.grid, events=e)
        self.assertEqual(d.ret.iloc[4], -1.)
        e["value"] = -1.1
        with self.assertRaises(ValueError):
            monthly_observations(self.price, self.grid, events=e)

    def test_post_delist_quote_is_not_a_return(self):
        d = monthly_observations(self.price, self.grid, delist_date="2020-03-31")
        self.assertTrue(np.isnan(d.ret.iloc[3]))
        self.assertEqual(d.return_status.iloc[3], "post_delist")


class TestPricingInterpretation(unittest.TestCase):
    def test_nonzero_spread_is_consistent_with_zero_alpha(self):
        x = pd.Series(np.linspace(-.05, .07, 60))
        y = .4 * x
        fit = ols_alpha_beta(y, x, 6)
        self.assertGreater(y.mean(), 0)
        self.assertAlmostEqual(fit["alpha"], 0, places=12)
        self.assertAlmostEqual(fit["beta"], .4, places=12)

    def test_test_window_threshold_is_explicit(self):
        x = pd.Series(np.arange(10, dtype=float))
        self.assertIsNone(ols_alpha_beta(x, x, 6, min_obs=12))


if __name__ == "__main__":
    unittest.main()
