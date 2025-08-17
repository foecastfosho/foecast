# forecast_model.py
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Iterable, Tuple, Dict, Optional, List
import numpy as np
import pandas as pd

# --------- Constants ----------
DAYS_PER_MONTH: float = 30.4375
EPS: float = 1e-6
MAX_INITIAL_DECLINE: float = 1.0   # 100%
MAX_TERMINAL_DECLINE: float = 0.30 # 30%

# --------- Data Model ----------
@dataclass(frozen=True)
class Well:
    """Production/decline parameters for a single well (rates are DAILY)."""
    name: str
    first_prod_date: pd.Timestamp
    qi_oil: float
    qi_gas: float
    qi_ngl: float
    initial_decline: float   # fraction/yr at t=0 (e.g., 0.75 = 75%/yr)
    b_factor: float          # hyperbolic b (0 => exponential)
    terminal_decline: float  # fraction/yr for exponential tail

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["first_prod_date"] = self.first_prod_date.isoformat()
        return d

    @staticmethod
    def from_dict(d: Dict) -> "Well":
        return Well(
            name=d["name"],
            first_prod_date=pd.to_datetime(d["first_prod_date"]),
            qi_oil=float(d.get("qi_oil", 0.0)),
            qi_gas=float(d.get("qi_gas", 0.0)),
            qi_ngl=float(d.get("qi_ngl", 0.0)),
            initial_decline=float(d.get("initial_decline", 0.0)),
            b_factor=float(d.get("b_factor", 0.0)),
            terminal_decline=float(d.get("terminal_decline", 0.1)),
        )

# --------- Decline Models ----------
def _hyp_decline(qi: float, di: float, b: float, t_years: np.ndarray) -> np.ndarray:
    """Hyperbolic decline; b=0 reduces to exponential."""
    if abs(b) < EPS:
        return qi * np.exp(-di * t_years)
    return qi / np.power((1.0 + b * di * t_years), (1.0 / b))

def _transition_to_exp(
    q_monthly: np.ndarray, di0: float, b: float, d_term: float, dt_years: float
) -> np.ndarray:
    """
    Given hyperbolic monthly rates, transition to exponential when
    instantaneous decline <= terminal decline.
    """
    q = q_monthly.copy()
    if d_term <= 0:
        return q

    # instantaneous decline for hyperbolic: D(t) = di0 / (1 + b*di0*t)
    t = np.arange(len(q)) * dt_years
    inst_decl = di0 / (1.0 + np.maximum(b, 0.0) * di0 * t)
    # first index when hyperbolic decline falls below terminal
    idx = int(np.argmax(inst_decl <= d_term)) if np.any(inst_decl <= d_term) else -1
    if idx > 0:
        # continue from q[idx] with exponential tail at d_term
        t_tail = np.arange(len(q) - idx) * dt_years
        q[idx:] = q[idx] * np.exp(-d_term * t_tail)
    return q

def well_monthly_rate_series(
    qi_daily: float, di_initial: float, b: float, d_term: float, months: int
) -> np.ndarray:
    """
    Return a MONTHLY average rate series (in daily-rate units) length=months.
    Decline is hyperbolic transitioning to exponential at terminal decline.
    """
    dt_years = 1.0 / 12.0
    t_years = np.arange(months) * dt_years
    q_daily = _hyp_decline(qi_daily, di_initial, b, t_years)
    q_daily = _transition_to_exp(q_daily, di_initial, b, d_term, dt_years)
    return np.maximum(q_daily, 0.0)

# --------- Price Deck ----------
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
) -> pd.DataFrame:
    """
    Build monthly price deck with columns ['date','oil','gas','ngl'].
    If custom_df is provided, it must contain those columns; otherwise, a geometric
    month-over-month growth deck is generated from the *_start and *_growth inputs.
    """
    idx = pd.date_range(pd.to_datetime(start_date), periods=months, freq="MS")
    if custom_df is not None and not custom_df.empty:
        df = custom_df.copy()
        # normalize date, ensure monthly start
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

# --------- Cash Flow Engine ----------
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
    """
    Calculate monthly gross volumes, realized prices, revenue, royalty, and net cash flow.

    Note:
    - Floors realized prices at 0 to avoid negative revenue from extreme differentials.
    - Applies royalty_decimal OR NRI (if provided). If both are provided, NRI takes precedence.
    """
    idx = pd.date_range(pd.to_datetime(start_date), periods=months, freq="MS")
    deck = price_deck.set_index("date").reindex(idx).fillna(method="ffill").reset_index()
    deck.rename(columns={"index": "date"}, inplace=True)

    oil_series = np.zeros(months)
    gas_series = np.zeros(months)
    ngl_series = np.zeros(months)

    for w in wells:
        # months active for this well (shift by first_prod_date)
        if pd.to_datetime(w.first_prod_date) > idx[-1]:
            continue
        offset = max(0, (pd.to_datetime(w.first_prod_date).to_period("M") - idx[0].to_period("M")).n)
        span = months - offset
        if span <= 0:
            continue

        qo = well_monthly_rate_series(w.qi_oil, w.initial_decline, w.b_factor, w.terminal_decline, span)
        qg = well_monthly_rate_series(w.qi_gas, w.initial_decline, w.b_factor, w.terminal_decline, span)
        qn = well_monthly_rate_series(w.qi_ngl, w.initial_decline, w.b_factor, w.terminal_decline, span)

        oil_series[offset:] += qo * DAYS_PER_MONTH  # daily → monthly volume
        gas_series[offset:] += qg * DAYS_PER_MONTH
        ngl_series[offset:] += qn * DAYS_PER_MONTH

    # Realized prices after differentials, transport, and post-prod (% of index price)
    oil_price = deck["oil"].to_numpy()
    gas_price = deck["gas"].to_numpy()
    ngl_price = deck["ngl"].to_numpy()

    net_oil_price = np.maximum(0.0, oil_price - oil_diff - transport_cost - (oil_price * post_prod_cost_pct))
    net_gas_price = np.maximum(0.0, gas_price - gas_diff - transport_cost - (gas_price * post_prod_cost_pct))
    net_ngl_price = np.maximum(0.0, ngl_price - ngl_diff - transport_cost - (ngl_price * post_prod_cost_pct))

    gross_rev = (oil_series * net_oil_price) + (gas_series * net_gas_price) + (ngl_series * net_ngl_price)

    # Taxes
    oil_tax = oil_series * net_oil_price * severance_tax_pct_oil
    gas_tax = gas_series * net_gas_price * severance_tax_pct_gas
    ngl_tax = ngl_series * net_ngl_price * severance_tax_pct_ngl
    taxes = oil_tax + gas_tax + ngl_tax

    # Interest share
    share = nri if nri is not None else royalty_decimal
    share = float(np.clip(share, 0.0, 1.0))

    # Cash flows (monthly)
    royalty_rev = gross_rev * share
    net_cash = royalty_rev - taxes - other_fixed_cost_per_month

    df = pd.DataFrame(
        {
            "date": idx,
            "oil_mcf_or_bbl": oil_series,
            "gas_mcf_or_bbl": gas_series,
            "ngl_bbl": ngl_series,
            "net_oil_price": net_oil_price,
            "net_gas_price": net_gas_price,
            "net_ngl_price": net_ngl_price,
            "gross_revenue": gross_rev,
            "taxes": taxes,
            "royalty_share": share,
            "royalty_revenue": royalty_rev,
            "net_cash_flow": net_cash,
        }
    )

    summary = {
        "PV0": float(np.sum(net_cash)),
        # PV10 etc. can be added by caller via discounting (see compute_investment_metrics)
    }
    return df, summary

# --------- Financial Metrics ----------
def _irr(cashflows: np.ndarray) -> Optional[float]:
    """Robust IRR fallback: try numpy_financial, then np.irr; return None if no sign change."""
    try:
        import numpy_financial as npf  # type: ignore
        return float(npf.irr(cashflows))
    except Exception:
        pass
    try:
        return float(np.irr(cashflows))  # deprecated but available on many systems
    except Exception:
        return None

def discounted(cashflows: np.ndarray, rate_annual: float) -> np.ndarray:
    """Discount monthly cash flows using an annual rate (converted to monthly)."""
    r_m = (1.0 + rate_annual) ** (1.0 / 12.0) - 1.0
    t = np.arange(len(cashflows))
    return cashflows / (1.0 + r_m) ** t

def compute_investment_metrics(
    monthly_net_cash: np.ndarray,
    acquisition_price: float,
    discount_rate_annual: float,
) -> Dict[str, Optional[float]]:
    """
    Key investment metrics for royalty/NRI buyers.
    Returns: IRR, Payback (months & years), Discounted Payback, NPV, PI, MOIC.
    """
    # IRR: initial outflow then inflows
    cfs_for_irr = np.concatenate(([-abs(acquisition_price)], monthly_net_cash))
    irr = _irr(cfs_for_irr)

    # NPV, PI
    pv_series = discounted(monthly_net_cash, discount_rate_annual)
    npv = float(np.sum(pv_series) - acquisition_price)
    moic = float((np.sum(monthly_net_cash)) / acquisition_price) if acquisition_price > 0 else None
    pi = float(np.sum(pv_series) / acquisition_price) if acquisition_price > 0 else None

    # Payback (undiscounted)
    cum = np.cumsum(monthly_net_cash) - acquisition_price
    payback_idx = int(np.argmax(cum >= 0)) if np.any(cum >= 0) else -1
    payback_months = payback_idx if payback_idx >= 0 else None

    # Discounted Payback
    cum_disc = np.cumsum(pv_series) - acquisition_price
    dpb_idx = int(np.argmax(cum_disc >= 0)) if np.any(cum_disc >= 0) else -1
    dpb_months = dpb_idx if dpb_idx >= 0 else None

    return {
        "IRR": irr,
        "NPV": npv,
        "MOIC": moic,
        "PI": pi,
        "PaybackMonths": payback_months,
        "PaybackYears": (payback_months / 12.0) if isinstance(payback_months, int) else None,
        "DiscountedPaybackMonths": dpb_months,
        "DiscountedPaybackYears": (dpb_months / 12.0) if isinstance(dpb_months, int) else None,
    }

# --------- Monte Carlo ----------
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
    vol_sigma: Tuple[float, float, float] = (0.1, 0.1, 0.1),    # oil, gas, ngl
    price_sigma: Tuple[float, float, float] = (0.15, 0.2, 0.2), # oil, gas, ngl
    rng_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simple MC: lognormal multipliers on qi (volumes) and prices.
    Returns a DataFrame of iteration metrics (IRR, NPV, MOIC, etc.).
    """
    rng = np.random.default_rng(rng_seed)
    oil_v, gas_v, ngl_v = vol_sigma
    oil_p, gas_p, ngl_p = price_sigma

    out: List[Dict[str, float]] = []
    for i in range(iterations):
        # Volume multipliers
        vm_o = rng.lognormal(mean=0.0, sigma=oil_v)
        vm_g = rng.lognormal(mean=0.0, sigma=gas_v)
        vm_n = rng.lognormal(mean=0.0, sigma=ngl_v)

        # Price multipliers (apply to deck)
        pm_o = rng.lognormal(mean=0.0, sigma=oil_p)
        pm_g = rng.lognormal(mean=0.0, sigma=gas_p)
        pm_n = rng.lognormal(mean=0.0, sigma=ngl_p)
        deck = base_deck.copy()
        deck["oil"] = deck["oil"] * pm_o
        deck["gas"] = deck["gas"] * pm_g
        deck["ngl"] = deck["ngl"] * pm_n

        # Perturb wells (clone safely)
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

        out.append(
            {
                "iteration": i + 1,
                "PV0": float(df["net_cash_flow"].sum()),
                # Keep metrics extensible; app can compute IRR/NPV given acquisition cost
            }
        )

    return pd.DataFrame(out)
