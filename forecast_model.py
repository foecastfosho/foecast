"""
Core forecast and financial modeling engine for the royalty DCF application.

This module encapsulates all business logic for decline curve forecasting,
price deck generation, cash flow computation, investment metrics, and
Monte Carlo simulations. By separating these functions from the Streamlit
interface, we facilitate testing and future reuse in other contexts.

The primary abstractions are:

* ``Well``: an immutable dataclass capturing production parameters.
* ``build_price_deck``: create a monthly price forecast, optionally from a custom
  DataFrame.
* ``compute_cash_flows``: generate per‑period cash flows for one or more wells
  given a price deck and tax assumptions.
* ``compute_investment_metrics``: derive IRR, NPV, payback periods, and other
  metrics from a cash flow vector and an acquisition cost.
* ``run_monte_carlo``: perform a simple Monte Carlo simulation over volume and
  price uncertainties.

Note that all rates (qi, decline rates, b‑factor) are expressed on an annual
basis and converted internally to monthly rates. NGL severance taxes are
assumed equal to oil severance taxes. Post‑production costs are applied as a
percentage of realized revenue. Net revenue interest (NRI) overrides the
royalty fraction if provided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, replace
from typing import Iterable, Optional, Tuple, Dict, List

__all__ = [
    "Well",
    "DAYS_PER_MONTH",
    "MAX_INITIAL_DECLINE",
    "MAX_TERMINAL_DECLINE",
    "build_price_deck",
    "compute_cash_flows",
    "compute_investment_metrics",
    "run_monte_carlo",
]

# -----------------------------------------------------------------------------
# Constants
#
# These constants centralize common values and bounds used throughout the
# forecasting engine. Adjust here if business rules change (e.g., maximum
# terminal decline).

# Average number of days per month used to convert daily rates to monthly
# volumes. Many decline curve analyses use 365.25/12 ≈ 30.4375 days.
DAYS_PER_MONTH: float = 30.4375

# Numerical epsilon used to detect effectively zero b‑factors.
EPS: float = 1e-6

# Maximum allowed annual decline rates for initial and terminal decline.
MAX_INITIAL_DECLINE: float = 1.0   # 100%
MAX_TERMINAL_DECLINE: float = 0.30 # 30%


# -----------------------------------------------------------------------------
# Data model

@dataclass(frozen=True)
class Well:
    """Immutable representation of a single well's decline parameters.

    Attributes
    ----------
    name : str
        Identifier for the well.
    first_prod_date : pd.Timestamp
        The first calendar date of production. Production prior to this date
        is ignored.
    qi_oil, qi_gas, qi_ngl : float
        Initial production rates (units per day) for oil, gas, and NGL.
    initial_decline : float
        Initial annual decline rate (fraction). For example, 0.75 for 75%/yr.
    b_factor : float
        Hyperbolic b‑factor. A value of 0 implies exponential decline.
    terminal_decline : float
        Terminal annual exponential decline (fraction). The decline curve
        transitions to exponential behaviour once the instantaneous decline
        falls below this terminal rate.
    royalty_decimal : float
        The royalty fraction for the well (e.g., 0.1875 for a 3/16th interest).
    nri : float
        Net revenue interest (1 minus burdens). If provided this will override
        the royalty fraction when computing cash flows.
    """

    name: str
    first_prod_date: pd.Timestamp
    qi_oil: float
    qi_gas: float
    qi_ngl: float
    initial_decline: float
    b_factor: float
    terminal_decline: float
    royalty_decimal: float
    nri: float

    def to_dict(self) -> Dict[str, float | str]:
        """Serialize the well to a JSON‑serializable dictionary."""
        d = asdict(self)
        d["first_prod_date"] = self.first_prod_date.isoformat()
        return d

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "Well":
        """Construct a Well from its dictionary representation."""
        return Well(
            name=data["name"],
            first_prod_date=pd.to_datetime(data["first_prod_date"]),
            qi_oil=float(data["qi_oil"]),
            qi_gas=float(data["qi_gas"]),
            qi_ngl=float(data["qi_ngl"]),
            initial_decline=float(data["initial_decline"]),
            b_factor=float(data["b_factor"]),
            terminal_decline=float(data["terminal_decline"]),
            royalty_decimal=float(data["royalty_decimal"]),
            nri=float(data["nri"]),
        )


# -----------------------------------------------------------------------------
# Decline models

def _hyperbolic_rate(qi: float, di: float, b: float, t: np.ndarray) -> np.ndarray:
    """Compute a hyperbolic decline curve on a per‐period basis.

    Parameters
    ----------
    qi : float
        Initial rate (units per day).
    di : float
        Initial annual decline rate (fraction).
    b : float
        Arps b‑factor. When b ≈ 0 the decline becomes exponential.
    t : ndarray
        Time array in years.

    Returns
    -------
    ndarray
        Rate series (units per day).
    """
    if abs(b) < EPS:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def _transition_to_exp(q: np.ndarray, di: float, b: float, d_term: float, dt: float) -> np.ndarray:
    """Apply exponential tail once instantaneous decline falls below terminal.

    Given a hyperbolic rate series, detect the period where the instantaneous
    decline rate becomes less than or equal to the terminal decline. Beyond
    that period, apply an exponential decline at d_term.

    Parameters
    ----------
    q : ndarray
        Hyperbolic rate series (units per day).
    di : float
        Initial decline (annual fraction).
    b : float
        Hyperbolic b‑factor.
    d_term : float
        Terminal decline (annual fraction).
    dt : float
        Time step in years between samples.

    Returns
    -------
    ndarray
        Rate series with an exponential tail applied.
    """
    if d_term <= 0:
        return q

    t = np.arange(len(q)) * dt
    inst_decl = di / (1.0 + np.maximum(b, 0.0) * di * t)
    indices = np.where(inst_decl <= d_term)[0]
    if len(indices) == 0:
        return q
    switch = indices[0]
    q_out = q.copy()
    q_switch = q_out[switch]
    tail_t = np.arange(len(q) - switch) * dt
    q_out[switch:] = q_switch * np.exp(-d_term * tail_t)
    return q_out


def well_rate_series(
    qi: float,
    initial_decline: float,
    b_factor: float,
    terminal_decline: float,
    months: int,
) -> np.ndarray:
    """Generate a monthly rate series for a single stream (oil, gas, or NGL).

    Parameters
    ----------
    qi : float
        Initial daily rate.
    initial_decline : float
        Annual decline rate at time zero.
    b_factor : float
        Arps b‑factor.
    terminal_decline : float
        Terminal annual decline for the exponential tail.
    months : int
        Number of months to forecast.

    Returns
    -------
    ndarray
        Monthly average rate series (units per day).
    """
    dt_years = 1.0 / 12.0
    t_years = np.arange(months) * dt_years
    q = _hyperbolic_rate(qi, initial_decline, b_factor, t_years)
    q = _transition_to_exp(q, initial_decline, b_factor, terminal_decline, dt_years)
    return np.maximum(q, 0.0)


# -----------------------------------------------------------------------------
# Price deck

def build_price_deck(
    start_date: pd.Timestamp,
    months: int,
    oil_start: float,
    oil_mom_growth: float,
    gas_start: float,
    gas_mom_growth: float,
    ngl_start: float,
    ngl_mom_growth: float,
    custom_df: Optional[pd.DataFrame] = None,
    uploaded_deck: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Construct a monthly price deck.

    If ``custom_df`` is provided, it must have columns ``date``, ``oil``,
    ``gas``, and ``ngl``. The function resamples this deck to monthly
    frequency via forward fill. Otherwise, a geometric price ramp is built
    from the start prices and month‑over‑month growth factors.

    Parameters
    ----------
    start_date : Timestamp
        First month in the deck.
    months : int
        Number of months to produce.
    oil_start, gas_start, ngl_start : float
        Starting price levels.
    oil_mom_growth, gas_mom_growth, ngl_mom_growth : float
        Per‑month growth rates (e.g., 0.01 for 1% increase per month).
    custom_df : DataFrame, optional
        Custom deck to override flat or ramped pricing.

    Returns
    -------
    DataFrame
        Monthly deck with columns ``date``, ``oil``, ``gas``, and ``ngl``.
    """
    idx = pd.date_range(pd.to_datetime(start_date), periods=months, freq="MS")
    # Support backward compatibility: prefer ``custom_df`` but also accept
    # legacy argument ``uploaded_deck``.  If both are provided, ``custom_df``
    # takes precedence.
    df = None
    if custom_df is not None:
        df = custom_df
    elif uploaded_deck is not None:
        df = uploaded_deck
    # Use the provided deck if available
    if df is not None and not df.empty:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
        df = (
            df.set_index("date")[["oil", "gas", "ngl"]]
            .reindex(idx, method="ffill")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        return df
    oil = oil_start * (1.0 + oil_mom_growth) ** np.arange(months)
    gas = gas_start * (1.0 + gas_mom_growth) ** np.arange(months)
    ngl = ngl_start * (1.0 + ngl_mom_growth) ** np.arange(months)
    return pd.DataFrame({"date": idx, "oil": oil, "gas": gas, "ngl": ngl})


# -----------------------------------------------------------------------------
# Cash flow engine

def compute_cash_flows(
    wells: Iterable[Well],
    start_date: pd.Timestamp,
    months: int,
    price_deck: pd.DataFrame,
    royalty_decimal: float,
    nri: Optional[float],
    severance_tax_pct_oil: float,
    severance_tax_pct_gas: float,
    severance_tax_pct_ngl: float,
    oil_diff: float,
    gas_diff: float,
    ngl_diff: float,
    transport_cost: float,
    post_prod_cost_pct: float,
    other_fixed_cost_per_month: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute monthly cash flows for one or more wells.

    This function aggregates production across all wells, applies price
    differentials, severance taxes, post‑production costs, and deducts a
    fixed monthly cost if provided. The revenue is then shared according
    to ``royalty_decimal`` or ``nri`` (the latter overrides the former).

    Parameters
    ----------
    wells : iterable of Well
        The wells to forecast.
    start_date : Timestamp
        First month of the forecast.
    months : int
        Number of months to forecast.
    price_deck : DataFrame
        Price deck with columns ``date``, ``oil``, ``gas``, ``ngl``.
    royalty_decimal : float
        Royalty fraction for revenue sharing. Ignored if ``nri`` is not None.
    nri : float, optional
        Net revenue interest. Overrides ``royalty_decimal`` when provided.
    severance_tax_pct_oil, severance_tax_pct_gas, severance_tax_pct_ngl : float
        Severance tax rates for each product.
    oil_diff, gas_diff, ngl_diff : float
        Price differentials subtracted from base prices.
    transport_cost : float
        Per‑unit transportation cost subtracted from all product prices.
    post_prod_cost_pct : float
        Fractional post‑production cost applied to gross revenue.
    other_fixed_cost_per_month : float, optional
        Additional fixed cost deducted each month.

    Returns
    -------
    DataFrame
        Monthly cash flow table with production volumes, realized prices,
        gross revenue, taxes, and net cash flow.
    dict
        Dictionary summarizing undiscounted and discounted cash flows.
    """
    idx = pd.date_range(pd.to_datetime(start_date), periods=months, freq="MS")
    deck = price_deck.set_index("date").reindex(idx).ffill().reset_index()
    deck.rename(columns={"index": "date"}, inplace=True)

    oil_vol = np.zeros(months)
    gas_vol = np.zeros(months)
    ngl_vol = np.zeros(months)

    for w in wells:
        # offset relative to start_date
        offset = max(0, int((w.first_prod_date.to_period("M") - idx[0].to_period("M")).n))
        span = months - offset
        if span <= 0:
            continue
        qo = well_rate_series(w.qi_oil, w.initial_decline, w.b_factor, w.terminal_decline, span)
        qg = well_rate_series(w.qi_gas, w.initial_decline, w.b_factor, w.terminal_decline, span)
        qn = well_rate_series(w.qi_ngl, w.initial_decline, w.b_factor, w.terminal_decline, span)
        oil_vol[offset:] += qo * DAYS_PER_MONTH
        gas_vol[offset:] += qg * DAYS_PER_MONTH
        ngl_vol[offset:] += qn * DAYS_PER_MONTH

    # realized prices
    price_o = deck["oil"].to_numpy() - oil_diff - transport_cost - deck["oil"].to_numpy() * post_prod_cost_pct
    price_g = deck["gas"].to_numpy() - gas_diff - transport_cost - deck["gas"].to_numpy() * post_prod_cost_pct
    price_n = deck["ngl"].to_numpy() - ngl_diff - transport_cost - deck["ngl"].to_numpy() * post_prod_cost_pct
    price_o = np.maximum(price_o, 0.0)
    price_g = np.maximum(price_g, 0.0)
    price_n = np.maximum(price_n, 0.0)

    rev = oil_vol * price_o + gas_vol * price_g + ngl_vol * price_n

    tax = (
        oil_vol * price_o * severance_tax_pct_oil
        + gas_vol * price_g * severance_tax_pct_gas
        + ngl_vol * price_n * severance_tax_pct_ngl
    )

    net_rev = rev - tax - other_fixed_cost_per_month
    # revenue share
    share = nri if nri is not None else royalty_decimal
    share = float(np.clip(share, 0.0, 1.0))
    net_cf = net_rev * share

    df = pd.DataFrame(
        {
            "date": idx,
            "oil_vol": oil_vol,
            "gas_vol": gas_vol,
            "ngl_vol": ngl_vol,
            "net_oil_price": price_o,
            "net_gas_price": price_g,
            "net_ngl_price": price_n,
            "gross_revenue": rev,
            "taxes": tax,
            "net_cash_flow": net_cf,
        }
    )

    summary = {
        "PV0": float(net_cf.sum()),
    }
    return df, summary


# -----------------------------------------------------------------------------
# Financial metrics

def _irr(cfs: np.ndarray) -> Optional[float]:
    """Compute IRR with fallback to numpy_financial or numpy.

    Returns None if IRR cannot be computed (e.g., no sign change).
    """
    try:
        import numpy_financial as npf  # type: ignore
        return float(npf.irr(cfs))
    except Exception:
        pass
    try:
        return float(np.irr(cfs))  # type: ignore
    except Exception:
        return None


def discounted(cfs: np.ndarray, rate_annual: float) -> np.ndarray:
    """Discount monthly cash flows using an annual discount rate."""
    r_m = (1.0 + rate_annual) ** (1.0 / 12.0) - 1.0
    t = np.arange(len(cfs))
    return cfs / (1.0 + r_m) ** t


def compute_investment_metrics(
    monthly_net_cash: np.ndarray,
    acquisition_price: float,
    discount_rate_annual: float,
) -> Dict[str, Optional[float]]:
    """Calculate IRR, NPV, payback periods, profitability index and MOIC.

    Parameters
    ----------
    monthly_net_cash : ndarray
        Cash inflows per month (length N).
    acquisition_price : float
        Upfront capital outlay. Treated as a negative cash flow at t=0.
    discount_rate_annual : float
        Annual discount rate used for NPV and discounted payback.

    Returns
    -------
    dict
        Dictionary with keys ``IRR``, ``NPV``, ``MOIC``, ``PI``,
        ``PaybackMonths``, ``PaybackYears``, ``DiscountedPaybackMonths``,
        ``DiscountedPaybackYears``.
    """
    cfs = np.concatenate(([-abs(acquisition_price)], monthly_net_cash))
    irr = _irr(cfs)
    pv_series = discounted(monthly_net_cash, discount_rate_annual)
    npv = float(pv_series.sum() - acquisition_price)
    moic = float(monthly_net_cash.sum() / acquisition_price) if acquisition_price > 0 else None
    pi = float(pv_series.sum() / acquisition_price) if acquisition_price > 0 else None
    cum = np.cumsum(monthly_net_cash) - acquisition_price
    pay_idx = int(np.argmax(cum >= 0)) if np.any(cum >= 0) else -1
    pay_months = pay_idx if pay_idx >= 0 else None
    cum_disc = np.cumsum(pv_series) - acquisition_price
    dpb_idx = int(np.argmax(cum_disc >= 0)) if np.any(cum_disc >= 0) else -1
    dpb_months = dpb_idx if dpb_idx >= 0 else None
    return {
        "IRR": irr,
        "NPV": npv,
        "MOIC": moic,
        "PI": pi,
        "PaybackMonths": pay_months,
        "PaybackYears": (pay_months / 12.0) if isinstance(pay_months, int) else None,
        "DiscountedPaybackMonths": dpb_months,
        "DiscountedPaybackYears": (dpb_months / 12.0) if isinstance(dpb_months, int) else None,
    }


# -----------------------------------------------------------------------------
# Monte Carlo

def run_monte_carlo(
    wells: Iterable[Well],
    start_date: pd.Timestamp,
    months: int,
    base_deck: pd.DataFrame,
    royalty_decimal: float,
    nri: Optional[float],
    severance_oil: float,
    severance_gas: float,
    severance_ngl: float,
    oil_diff: float,
    gas_diff: float,
    ngl_diff: float,
    transport: float,
    post_prod_cost_pct: float,
    iterations: int = 200,
    vol_sigma: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    price_sigma: Tuple[float, float, float] = (0.15, 0.2, 0.2),
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Run a Monte Carlo simulation on price and volume uncertainty.

    Prices and volumes are perturbed by lognormal random multipliers. For each
    simulation, new wells and price decks are created and cash flows computed.

    Returns
    -------
    DataFrame
        Summary metrics (currently PV0 only) per iteration.
    """
    rng = np.random.default_rng(rng_seed)
    sigma_v_o, sigma_v_g, sigma_v_n = vol_sigma
    sigma_p_o, sigma_p_g, sigma_p_n = price_sigma
    results: List[Dict[str, float]] = []
    for i in range(iterations):
        vm_o = rng.lognormal(mean=0.0, sigma=sigma_v_o)
        vm_g = rng.lognormal(mean=0.0, sigma=sigma_v_g)
        vm_n = rng.lognormal(mean=0.0, sigma=sigma_v_n)
        pm_o = rng.lognormal(mean=0.0, sigma=sigma_p_o)
        pm_g = rng.lognormal(mean=0.0, sigma=sigma_p_g)
        pm_n = rng.lognormal(mean=0.0, sigma=sigma_p_n)
        deck = base_deck.copy()
        deck["oil"] = deck["oil"] * pm_o
        deck["gas"] = deck["gas"] * pm_g
        deck["ngl"] = deck["ngl"] * pm_n
        perturbed: List[Well] = []
        for w in wells:
            perturbed.append(
                replace(
                    w,
                    qi_oil=max(0.0, w.qi_oil * vm_o),
                    qi_gas=max(0.0, w.qi_gas * vm_g),
                    qi_ngl=max(0.0, w.qi_ngl * vm_n),
                )
            )
        df, _ = compute_cash_flows(
            wells=perturbed,
            start_date=start_date,
            months=months,
            price_deck=deck,
            royalty_decimal=royalty_decimal,
            nri=nri,
            severance_tax_pct_oil=severance_oil,
            severance_tax_pct_gas=severance_gas,
            severance_tax_pct_ngl=severance_ngl,
            oil_diff=oil_diff,
            gas_diff=gas_diff,
            ngl_diff=ngl_diff,
            transport_cost=transport,
            post_prod_cost_pct=post_prod_cost_pct,
        )
        results.append({"iteration": i + 1, "PV0": float(df["net_cash_flow"].sum())})
    return pd.DataFrame(results)