import pytest
import numpy as np
import pandas as pd
from price_simulator import PowerPrices, compute_stats, settlement_period


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def pp():
    return PowerPrices()

@pytest.fixture
def one_week_prices(pp):
    return pp.generate_price_curve('2026-01-01', '2026-01-07')



# ══════════════════════════════════════════════════════════════════════════════
# LOW LEVEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSettlementPeriod:
    def test_midnight_is_sp1(self):
        idx = pd.DatetimeIndex(['2026-01-01 00:00'])
        assert settlement_period(idx)[0] == 1

    def test_half_past_midnight_is_sp2(self):
        idx = pd.DatetimeIndex(['2026-01-01 00:30'])
        assert settlement_period(idx)[0] == 2

    def test_last_sp_of_day_is_48(self):
        idx = pd.DatetimeIndex(['2026-01-01 23:30'])
        assert settlement_period(idx)[0] == 48

    def test_returns_numpy_array(self):
        idx = pd.date_range('2026-01-01', periods=48, freq='30min')
        result = settlement_period(idx)
        assert isinstance(result, np.ndarray)

    def test_length_matches_index(self):
        idx = pd.date_range('2026-01-01', periods=96, freq='30min')
        assert len(settlement_period(idx)) == 96


class TestDeepUpdate:
    def test_overrides_nested_value(self):
        result = PowerPrices.deep_update(
            {"ou": {"theta": 4.0, "sigma_mult": 5}},
            {"ou": {"theta": 7.0}}
        )
        assert result["ou"]["theta"] == 7.0
        assert result["ou"]["sigma_mult"] == 5  # untouched

    def test_does_not_mutate_default(self):
        default = {"base_price": 70}
        PowerPrices.deep_update(default, {"base_price": 100})
        assert default["base_price"] == 70

    def test_adds_new_key(self):
        result = PowerPrices.deep_update({"a": 1}, {"b": 2})
        assert result["b"] == 2


class TestComputeStats:
    def test_returns_all_keys(self):
        s = pd.Series([10.0, 20.0, 30.0])
        stats = compute_stats(s)
        assert set(stats.keys()) == {"max", "min", "std", "median", "mean"}

    def test_correct_values(self):
        s = pd.Series([10.0, 20.0, 30.0])
        stats = compute_stats(s)
        assert stats["max"] == 30.0
        assert stats["min"] == 10.0
        assert stats["mean"] == 20.0
        assert stats["median"] == 20.0


class TestIntradayCurve:
    def test_peak_near_expected_settlement_periods(self, pp):
        sp = np.arange(1, 49)
        ones = np.ones(48)
        curve = pp.intraday_curve(sp, ones, ones * 4, ones * 4)
        # Peaks should be near SP 16 and SP 37 per config
        assert curve[15] > curve[0]   # SP 16 higher than midnight
        assert curve[36] > curve[0]   # SP 37 higher than midnight

    def test_weekend_lower_than_weekday(self, pp):
        sp = np.arange(1, 49)
        wkd = pp.intraday_curve(sp, np.ones(48), np.ones(48) * 4, np.ones(48) * 4)
        we  = pp.intraday_curve(sp, np.ones(48) * 0.6, np.ones(48) * 4, np.ones(48) * 4)
        assert we.sum() < wkd.sum()


class TestSeasonality:
    def test_winter_higher_than_summer(self, pp):
        winter = pd.date_range('2026-01-01', periods=48, freq='30min')
        summer = pd.date_range('2026-07-01', periods=48, freq='30min')
        winter_curve = pp.seasonality_curve(winter.day_of_year)
        summer_curve = pp.seasonality_curve(summer.day_of_year)
        assert winter_curve.mean() > summer_curve.mean()

    def test_output_length_matches_input(self, pp):
        idx = pd.date_range('2026-01-01', periods=100, freq='30min')
        result = pp.seasonality_curve(idx.day_of_year)
        assert len(result) == 100


class TestOrnsteinUhlenbeck:
    def test_output_length_matches_input(self, pp):
        det = pd.Series(np.ones(96) * 70)
        result = pp.ornstein_uhlenbeck(det)
        assert len(result) == 96

    def test_mean_reverts_to_deterministic_curve(self, pp):
        # With high theta, OU should track det_curve closely
        pp.config['ou']['theta'] = 100.0
        pp.config['ou']['sigma_mult'] = 0.0
        det = pd.Series(np.ones(480) * 70)
        result = pp.ornstein_uhlenbeck(det)
        assert abs(result.mean() - 70) < 2.0


class TestWeekdayWeekend:
    def test_weekend_mult_lower(self, pp):
        idx = pd.date_range('2026-01-03', periods=48, freq='30min')  # Saturday
        mult, _, _ = pp.week_day_end(idx)
        assert all(mult == pp.config['weekday']['we_mult'])

    def test_weekday_mult_correct(self, pp):
        idx = pd.date_range('2026-01-05', periods=48, freq='30min')  # Monday
        mult, _, _ = pp.week_day_end(idx)
        assert all(mult == pp.config['weekday']['wkd_mult'])


# ══════════════════════════════════════════════════════════════════════════════
# HIGH LEVEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneratePriceCurve:
    def test_returns_dataframe(self, one_week_prices):
        assert isinstance(one_week_prices, pd.DataFrame)

    def test_has_correct_columns(self, one_week_prices):
        assert set(one_week_prices.columns) == {'datetime', 'power_prices'}

    def test_correct_number_of_rows(self, one_week_prices):
        # 6 days * 48 half-hour periods = 288 (date_range is end-inclusive)
        assert len(one_week_prices) == 288+1

    def test_datetime_column_is_30min_frequency(self, one_week_prices):
        diffs = one_week_prices['datetime'].diff().dropna()
        assert (diffs == pd.Timedelta('30min')).all()

    def test_prices_are_numeric(self, one_week_prices):
        assert pd.api.types.is_float_dtype(one_week_prices['power_prices'])

    def test_negative_prices_are_valid(self):
        # Run many curves and check negative prices can occur (not clipped)
        pp = PowerPrices(config={"ou": {"sigma_mult": 100}, "base_price": 10})
        prices = pd.concat([
            pp.generate_price_curve('2026-01-01', '2026-12-31')['power_prices']
            for _ in range(5)
        ])
        assert (prices < 0).any(), "Expected some negative prices across 5 yearly curves"

    def test_two_calls_produce_different_prices(self, pp):
        p1 = pp.generate_price_curve('2026-01-01', '2026-01-07')['power_prices']
        p2 = pp.generate_price_curve('2026-01-01', '2026-01-07')['power_prices']
        assert not p1.equals(p2)

    def test_two_calls_same_seed_produce_same_prices(self):
        pp=PowerPrices(config={'ou':{'seed':42}})
        p1 = pp.generate_price_curve('2026-01-01', '2026-01-07')['power_prices']
        p2 = pp.generate_price_curve('2026-01-01', '2026-01-07')['power_prices']
        assert p1.equals(p2)

    def test_invalid_date_range_raises_error(self, pp):
        with pytest.raises((ValueError, Exception)):
            pp.generate_price_curve('2026-12-31', '2026-01-01')

    def test_curve_stored_in_curve_dict(self, pp):
        pp.generate_price_curve('2026-01-01', '2026-01-07')
        assert len(pp.curve_dict) == 1

    def test_custom_config_affects_base_price(self):
        pp_high = PowerPrices(config={"base_price": 200})
        pp_low  = PowerPrices(config={"base_price": 10})
        high = pp_high.generate_price_curve('2026-01-01', '2026-01-31')['power_prices'].mean()
        low  = pp_low.generate_price_curve('2026-01-01', '2026-01-31')['power_prices'].mean()
        assert high > low


class TestAnalyseCurves:
    def test_returns_stats_for_each_curve(self, pp):
        pp.generate_price_curve('2026-01-01', '2026-01-07')
        pp.generate_price_curve('2026-02-01', '2026-02-07')
        stats = pp.analyse_curves()
        assert len(stats) == 2

    def test_stats_contain_expected_keys(self, pp):
        pp.generate_price_curve('2026-01-01', '2026-01-07')
        stats = pp.analyse_curves()
        for curve_stats in stats.values():
            assert set(curve_stats.keys()) == {"max", "min", "std", "median", "mean"}

    def test_max_greater_than_min(self, pp):
        pp.generate_price_curve('2026-01-01', '2026-01-07')
        stats = pp.analyse_curves()
        for curve_stats in stats.values():
            assert curve_stats['max'] > curve_stats['min']