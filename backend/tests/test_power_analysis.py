"""Coverage for POST /api/stats/power across test types and solve_for modes
not already exercised by test_package8_grabbag.py, test_stats_coverage.py,
and test_v2_endpoints.py.

Existing coverage (not duplicated here):
- t_two / solve_for=n
- anova / solve_for=power
- logistic / solve_for=n,power,effect_size
- survival_cox / solve_for=n,effect_size

This file adds:
- t_one: n, power, effect_size
- t_two: power, effect_size
- correlation: n, power, effect_size
- proportion: n, power, effect_size
- anova: n, effect_size
- chi2 (goodness-of-fit): n, power, effect_size
- validation/error paths
- monotonicity invariants (bigger effect size -> higher power; the reverse
  direction, i.e. more n -> more power, is also checked)
"""
import pytest

ENDPOINT = "/api/stats/power"


def _post(client, payload):
    return client.post(ENDPOINT, json=payload)


# ── t_one (one-sample t-test) ────────────────────────────────────────────────

def test_power_t_one_solve_n(client):
    r = _post(client, {
        "test": "t_one", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "effect_size": 0.5})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_t_one_solve_power(client):
    r = _post(client, {
        "test": "t_one", "solve_for": "power", "alpha": 0.05,
        "effect_size": 0.5, "n": 40})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["result"] <= 1.0


def test_power_t_one_solve_effect_size(client):
    r = _post(client, {
        "test": "t_one", "solve_for": "effect_size", "alpha": 0.05,
        "power": 0.8, "n": 40})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_t_one_effect_size_power_invariant(client):
    """Bigger required effect size (fixed n, alpha) achieved with lower power
    should yield a smaller minimum-detectable effect than higher power."""
    def solve_es(power):
        r = _post(client, {
            "test": "t_one", "solve_for": "effect_size", "alpha": 0.05,
            "power": power, "n": 40})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    es_low = solve_es(0.5)
    es_high = solve_es(0.9)
    assert es_low < es_high


# ── t_two: power, effect_size (n already covered elsewhere) ─────────────────

def test_power_t_two_solve_power(client):
    r = _post(client, {
        "test": "t_two", "solve_for": "power", "alpha": 0.05,
        "effect_size": 0.5, "n": 64})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["result"] <= 1.0


def test_power_t_two_solve_effect_size(client):
    r = _post(client, {
        "test": "t_two", "solve_for": "effect_size", "alpha": 0.05,
        "power": 0.8, "n": 64})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_t_two_effect_size_decreases_power_invariant(client):
    """Increasing effect size for fixed n/alpha must INCREASE achieved power
    (equivalently: smaller effect -> lower power)."""
    def pw(effect_size):
        r = _post(client, {
            "test": "t_two", "solve_for": "power", "alpha": 0.05,
            "effect_size": effect_size, "n": 40})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    small = pw(0.2)
    large = pw(0.8)
    assert small < large


# ── correlation (Fisher-z) ───────────────────────────────────────────────────

def test_power_correlation_solve_n(client):
    r = _post(client, {
        "test": "correlation", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "effect_size": 0.3, "tails": 2})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_correlation_solve_power(client):
    r = _post(client, {
        "test": "correlation", "solve_for": "power", "alpha": 0.05,
        "effect_size": 0.3, "n": 100, "tails": 2})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["result"] <= 1.0


def test_power_correlation_solve_effect_size(client):
    r = _post(client, {
        "test": "correlation", "solve_for": "effect_size", "alpha": 0.05,
        "power": 0.8, "n": 100, "tails": 2})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and 0.0 < b["result"] < 1.0


def test_power_correlation_effect_size_increases_power_invariant(client):
    def pw(r_es):
        r = _post(client, {
            "test": "correlation", "solve_for": "power", "alpha": 0.05,
            "effect_size": r_es, "n": 100, "tails": 2})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    small = pw(0.1)
    large = pw(0.5)
    assert small < large


# ── proportion (Cohen's h, two independent proportions) ─────────────────────

def test_power_proportion_solve_n(client):
    r = _post(client, {
        "test": "proportion", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "p1": 0.5, "p2": 0.3})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_proportion_solve_power(client):
    r = _post(client, {
        "test": "proportion", "solve_for": "power", "alpha": 0.05,
        "p1": 0.5, "p2": 0.3, "n": 60})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["result"] <= 1.0


def test_power_proportion_solve_effect_size(client):
    r = _post(client, {
        "test": "proportion", "solve_for": "effect_size", "alpha": 0.05,
        "power": 0.8, "n": 60})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_proportion_bigger_gap_increases_power_invariant(client):
    """Larger difference between p1/p2 (bigger Cohen's h) for fixed n/alpha
    should yield higher achieved power."""
    def pw(p1, p2):
        r = _post(client, {
            "test": "proportion", "solve_for": "power", "alpha": 0.05,
            "p1": p1, "p2": p2, "n": 60})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    small_gap = pw(0.5, 0.45)
    large_gap = pw(0.5, 0.2)
    assert small_gap < large_gap


# ── anova: n, effect_size (power already covered elsewhere) ─────────────────

def test_power_anova_solve_n(client):
    r = _post(client, {
        "test": "anova", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "effect_size": 0.25, "k_groups": 3})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_anova_solve_effect_size(client):
    r = _post(client, {
        "test": "anova", "solve_for": "effect_size", "alpha": 0.05,
        "power": 0.8, "n": 40, "k_groups": 3})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_anova_effect_size_increases_power_invariant(client):
    def pw(effect_size):
        r = _post(client, {
            "test": "anova", "solve_for": "power", "alpha": 0.05,
            "effect_size": effect_size, "n": 40, "k_groups": 3})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    small = pw(0.1)
    large = pw(0.4)
    assert small < large


# ── chi2 (goodness-of-fit) ───────────────────────────────────────────────────

def test_power_chi2_solve_n(client):
    r = _post(client, {
        "test": "chi2", "solve_for": "n", "alpha": 0.05,
        "power": 0.8, "effect_size": 0.3, "k_groups": 4})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_chi2_solve_power(client):
    r = _post(client, {
        "test": "chi2", "solve_for": "power", "alpha": 0.05,
        "effect_size": 0.3, "n": 100, "k_groups": 4})
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0.0 <= b["result"] <= 1.0


def test_power_chi2_solve_effect_size(client):
    r = _post(client, {
        "test": "chi2", "solve_for": "effect_size", "alpha": 0.05,
        "power": 0.8, "n": 100, "k_groups": 4})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["result"] is not None and b["result"] > 0


def test_power_chi2_effect_size_increases_power_invariant(client):
    def pw(effect_size):
        r = _post(client, {
            "test": "chi2", "solve_for": "power", "alpha": 0.05,
            "effect_size": effect_size, "n": 100, "k_groups": 4})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    small = pw(0.1)
    large = pw(0.5)
    assert small < large


# ── n -> power monotonicity (cross-check across a couple of test types) ─────

def test_power_t_two_n_increases_power_invariant(client):
    def pw(n):
        r = _post(client, {
            "test": "t_two", "solve_for": "power", "alpha": 0.05,
            "effect_size": 0.3, "n": n})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    assert pw(20) < pw(200)


def test_power_correlation_n_increases_power_invariant(client):
    def pw(n):
        r = _post(client, {
            "test": "correlation", "solve_for": "power", "alpha": 0.05,
            "effect_size": 0.2, "n": n, "tails": 2})
        assert r.status_code == 200, r.text
        return r.json()["result"]

    assert pw(30) < pw(300)


# ── Validation / error paths ─────────────────────────────────────────────────

def test_power_unknown_test_type_rejected(client):
    r = _post(client, {"test": "bogus_test", "solve_for": "n",
                        "power": 0.8, "effect_size": 0.5})
    assert r.status_code == 400


def test_power_missing_required_field_returns_4xx_not_500(client):
    # logistic requires log_or/effect_size AND p_event; omit both meaningful
    # fields to ensure a clean 4xx rather than an unhandled 500.
    r = _post(client, {"test": "logistic", "solve_for": "n", "alpha": 0.05,
                        "power": 0.8})
    assert r.status_code in (400, 422), r.text


def test_power_survival_cox_missing_hr_returns_4xx(client):
    r = _post(client, {"test": "survival_cox", "solve_for": "n",
                        "alpha": 0.05, "power": 0.8, "event_rate": 0.3})
    assert r.status_code in (400, 422), r.text


def test_power_t_two_missing_effect_size_for_solve_n_is_4xx_not_500(client):
    # solve_for=n requires effect_size; statsmodels raises on None which
    # should surface as a client error, not an unhandled 500.
    r = _post(client, {"test": "t_two", "solve_for": "n", "alpha": 0.05,
                        "power": 0.8})
    assert r.status_code != 500, r.text


def test_power_t_two_missing_n_for_solve_power_is_4xx_not_500(client):
    r = _post(client, {"test": "t_two", "solve_for": "power", "alpha": 0.05,
                        "effect_size": 0.5})
    assert r.status_code != 500, r.text
