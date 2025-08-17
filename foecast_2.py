import datetime
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Import improved forecasting engine from forecast_model. These functions
# encapsulate the core business logic for decline forecasting, price deck
# generation, cash flow computation, and investment metrics. By using
# them here, we ensure that the UI leverages the latest model
# enhancements (e.g. b‑factor handling, negative price clamping, etc.).
from forecast_model import (
    Well as FMWell,
    build_price_deck as fm_build_price_deck,
    compute_cash_flows,
    compute_investment_metrics,
)

# The Streamlit import is wrapped in a try/except so that the module can be
# imported for unit testing even if Streamlit is not installed.  If you run
# this file as an app (via `streamlit run`), Streamlit must be installed.
try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None


@dataclass
class Well:
    """Container for well properties used in the DCF model."""
    name: str
    well_type: str  # 'PDP', 'PUD', or 'Future'
    first_prod_date: datetime.date
    # Decline parameters
    qi_oil: float  # initial oil rate in bbl/d
    qi_gas: float  # initial gas rate in Mcf/d
    qi_ngl: float  # initial NGL rate in bbl/d
    b_factor: float
    initial_decline: float  # annual fraction (e.g., 0.7 for 70 %)
    terminal_decline: float  # annual fraction (e.g., 0.05 for 5 %)
    royalty_decimal: float  # fraction of royalty interest owned
    nri: float  # net revenue interest (1 - burdens)
    # Derived fields for internal use
    start_month_index: int = field(init=False, default=0)

    def set_start_index(self, model_start: datetime.date) -> None:
        """Calculate the zero‑based month index of the well’s first production.

        This method is called after all wells are defined so that the model can
        align production with the global time axis.
        """
        delta = (self.first_prod_date.year - model_start.year) * 12 + (
            self.first_prod_date.month - model_start.month
        )
        self.start_month_index = max(0, delta)


def hyperbolic_decline_rate(qi: float, Di: float, b: float, Dt: float, months: int) -> np.ndarray:
    """Generate a monthly production forecast using a hyperbolic decline curve.

    Parameters
    ----------
    qi : float
        Initial production rate (units per day).
    Di : float
        Initial decline rate per year (fraction).  For example, 0.7 means
        70 % decline in the first year.
    b : float
        Arps b‑factor.  Typical shale wells fall between 0.3 and 1.0; we use
        0.5 as a conservative default【787064351170111†L75-L87】.
    Dt : float
        Terminal exponential decline rate per year (fraction).  Once the
        hyperbolic decline rate falls below Dt, the curve transitions to
        exponential decline.
    months : int
        Number of months to forecast.

    Returns
    -------
    np.ndarray
        Array of monthly average production rates (units per month).  Units of
        the returned values are the same as `qi` multiplied by days per month.
    """
    # Convert annual rates to monthly rates
    Di_m = Di / 12.0
    Dt_m = Dt / 12.0
    t = np.arange(months, dtype=float)
    # Hyperbolic decline until the instantaneous decline rate falls below Dt
    # q(t) = qi / (1 + b * Di_m * t)**(1/b)
    q_hyp = qi / np.power(1 + b * Di_m * t, 1 / b)
    # Instantaneous decline rate d(t) = Di / (1 + b * Di_m * t)
    inst_decline = Di / (1 + b * Di_m * t)
    # Find the transition month index where inst_decline <= Dt
    switch_idx = np.argmax(inst_decline <= Dt)
    # If the hyperbolic decline never falls below the terminal rate, use all
    # hyperbolic values.  Otherwise, switch to exponential after the switch_idx.
    if switch_idx == 0 and inst_decline[0] <= Dt:
        # Start immediately on exponential decline
        q = q_hyp.copy()
        q = q * np.power(1 - Dt_m, t)  # exponential decline from initial rate
    elif inst_decline[-1] > Dt:
        # Hyperbolic curve never meets terminal decline; use entire hyperbolic
        q = q_hyp.copy()
    else:
        # Combine hyperbolic up to switch_idx, exponential afterwards
        q = q_hyp.copy()
        q[switch_idx:] = q_hyp[switch_idx] * np.power(1 - Dt_m, t[: months - switch_idx])
    # Convert daily rate to monthly volume (approximate 30 days per month)
    return q * 30.0


def build_net_price_deck(
    start_date: datetime.date,
    months: int,
    oil_price: float,
    gas_price: float,
    ngl_price: float,
    diff_oil: float,
    diff_gas: float,
    diff_ngl: float,
    uploaded_deck: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Create a monthly **net** price deck for oil, gas, and NGL.

    This helper generates a price deck where price differentials are applied
    directly to the input index prices.  It is retained for backward
    compatibility with portions of the code that compute cash flows using
    legacy logic.  If a custom deck is uploaded, its price columns are
    forward‑filled and differentials are applied to the resulting series.

    Parameters
    ----------
    start_date : datetime.date
        First date of the forecast horizon.
    months : int
        Number of months to generate.
    oil_price, gas_price, ngl_price : float
        Starting index prices for each commodity.
    diff_oil, diff_gas, diff_ngl : float
        Price differentials to subtract from index prices to get net prices.
    uploaded_deck : DataFrame, optional
        User‑supplied deck with columns ``date``, ``oil``, ``gas``, ``ngl``.

    Returns
    -------
    DataFrame
        Net price deck with columns ``date``, ``oil``, ``gas``, ``ngl``.
    """
    dates = [start_date + datetime.timedelta(days=30 * i) for i in range(months)]
    if uploaded_deck is not None:
        # Clean and sort the uploaded deck
        deck = uploaded_deck.copy()
        # Ensure correct dtypes
        for col in ['oil', 'gas', 'ngl']:
            deck[col] = pd.to_numeric(deck[col], errors='coerce')
        deck['date'] = pd.to_datetime(deck['date'])
        deck = deck.sort_values('date')
        deck = deck.reset_index(drop=True)
        # Build arrays of prices; extend the last available price if necessary
        oil_prices = []
        gas_prices = []
        ngl_prices = []
        last_oil = oil_price
        last_gas = gas_price
        last_ngl = ngl_price
        for dt in dates:
            mask = deck['date'] <= dt
            if mask.any():
                last_row = deck.loc[mask].iloc[-1]
                if not pd.isna(last_row['oil']):
                    last_oil = last_row['oil']
                if not pd.isna(last_row['gas']):
                    last_gas = last_row['gas']
                if not pd.isna(last_row['ngl']):
                    last_ngl = last_row['ngl']
            oil_prices.append(last_oil)
            gas_prices.append(last_gas)
            ngl_prices.append(last_ngl)
    else:
        oil_prices = [oil_price] * months
        gas_prices = [gas_price] * months
        ngl_prices = [ngl_price] * months
    # Apply price differentials
    oil_prices = np.array(oil_prices) - diff_oil
    gas_prices = np.array(gas_prices) - diff_gas
    ngl_prices = np.array(ngl_prices) - diff_ngl
    return pd.DataFrame(
        {
            'date': dates,
            'oil': oil_prices,
            'gas': gas_prices,
            'ngl': ngl_prices,
        }
    )


def compute_cash_flows(
    wells: List[Well],
    price_deck: pd.DataFrame,
    severance_tax_oil: float,
    severance_tax_gas: float,
    ad_valorem_tax: float,
    post_prod_cost_pct: float,
    discount_rate: float,
    forecast_months: int,
) -> pd.DataFrame:
    """Compute monthly cash flows and summary metrics for a list of wells.

    Parameters
    ----------
    wells : List[Well]
        List of wells with defined decline parameters and royalty interest info.
    price_deck : pd.DataFrame
        DataFrame with columns 'date', 'oil', 'gas', 'ngl' representing the
        monthly price deck net of differentials.
    severance_tax_oil : float
        Oil severance tax rate (e.g., 0.046 for 4.6 %).
    severance_tax_gas : float
        Natural gas severance tax rate (e.g., 0.075 for 7.5 %).
    ad_valorem_tax : float
        Ad valorem (property) tax rate (applied to gross revenue).
    post_prod_cost_pct : float
        Post‑production cost as a fraction of revenue (e.g., 0.02 for 2 %).
    discount_rate : float
        Annual discount rate (decimal) used to compute present value.
    forecast_months : int
        Number of months to forecast.

    Returns
    -------
    DataFrame
        A DataFrame containing monthly cash flows for each well and totals,
        plus additional columns for discounted cash flows and cumulative cash
        flows.
    """
    # Initialize arrays for aggregate production and revenue
    oil_vol = np.zeros(forecast_months)
    gas_vol = np.zeros(forecast_months)
    ngl_vol = np.zeros(forecast_months)
    net_royalty_volume = np.zeros(forecast_months)
    # For PV by well type, accumulate net cash by type
    pv_by_type = {'PDP': 0.0, 'PUD': 0.0, 'Future': 0.0}
    undiscounted_by_type = {'PDP': 0.0, 'PUD': 0.0, 'Future': 0.0}
    # Precompute monthly discount factor from annual discount rate
    disc_m = (1 + discount_rate) ** (1 / 12.0)
    discount_factors = 1 / np.power(disc_m, np.arange(forecast_months))
    # Loop over wells
    for well in wells:
        # Production arrays for this well
        oil = hyperbolic_decline_rate(
            qi=well.qi_oil,
            Di=well.initial_decline,
            b=well.b_factor,
            Dt=well.terminal_decline,
            months=forecast_months,
        )
        gas = hyperbolic_decline_rate(
            qi=well.qi_gas,
            Di=well.initial_decline,
            b=well.b_factor,
            Dt=well.terminal_decline,
            months=forecast_months,
        )
        ngl = hyperbolic_decline_rate(
            qi=well.qi_ngl,
            Di=well.initial_decline,
            b=well.b_factor,
            Dt=well.terminal_decline,
            months=forecast_months,
        )
        # Shift production by start month index
        oil_shift = np.zeros(forecast_months)
        gas_shift = np.zeros(forecast_months)
        ngl_shift = np.zeros(forecast_months)
        idx = well.start_month_index
        if idx < forecast_months:
            length = min(forecast_months - idx, forecast_months)
            oil_shift[idx : idx + length] = oil[:length]
            gas_shift[idx : idx + length] = gas[:length]
            ngl_shift[idx : idx + length] = ngl[:length]
        # Accumulate gross volumes
        oil_vol += oil_shift
        gas_vol += gas_shift
        ngl_vol += ngl_shift
        # Compute gross revenue for this well
        revenue_oil = oil_shift * price_deck['oil'].values
        revenue_gas = gas_shift * price_deck['gas'].values
        revenue_ngl = ngl_shift * price_deck['ngl'].values
        gross_rev = revenue_oil + revenue_gas + revenue_ngl
        # Taxes
        tax = (
            revenue_oil * severance_tax_oil
            + revenue_gas * severance_tax_gas
            + revenue_ngl * severance_tax_oil  # assume NGL taxed like oil
        )
        ad_val = gross_rev * ad_valorem_tax
        # Post‑production costs
        post_prod = gross_rev * post_prod_cost_pct
        # Net revenue after taxes and costs
        net_rev = gross_rev - tax - ad_val - post_prod
        # Net to royalty owner (royalty decimal * NRI)
        net_royalty = net_rev * well.royalty_decimal * well.nri
        # Accumulate for PV by type
        undiscounted_by_type[well.well_type] += net_royalty.sum()
        pv_by_type[well.well_type] += (net_royalty * discount_factors).sum()
        # Aggregate net royalty volume for weighting average royalty if needed
        net_royalty_volume += net_royalty
    # Aggregate cash flow across all wells (per month)
    gross_rev_total = (
        oil_vol * price_deck['oil'].values
        + gas_vol * price_deck['gas'].values
        + ngl_vol * price_deck['ngl'].values
    )
    tax_total = (
        oil_vol * price_deck['oil'].values * severance_tax_oil
        + gas_vol * price_deck['gas'].values * severance_tax_gas
        + ngl_vol * price_deck['ngl'].values * severance_tax_oil
    )
    ad_val_total = gross_rev_total * ad_valorem_tax
    post_prod_total = gross_rev_total * post_prod_cost_pct
    net_rev_total = gross_rev_total - tax_total - ad_val_total - post_prod_total
    # Weighted average royalty fraction across wells
    # Sum of (royalty * NRI * volume) divided by total volume
    total_volume = oil_vol + gas_vol + ngl_vol
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_roy_frac = np.sum([well.royalty_decimal * well.nri * (
            hyperbolic_decline_rate(well.qi_oil, well.initial_decline, well.b_factor, well.terminal_decline, forecast_months)
            + hyperbolic_decline_rate(well.qi_gas, well.initial_decline, well.b_factor, well.terminal_decline, forecast_months)
            + hyperbolic_decline_rate(well.qi_ngl, well.initial_decline, well.b_factor, well.terminal_decline, forecast_months)
        ) for well in wells], axis=0)
        if total_volume.sum() > 0:
            avg_roy_fraction = avg_roy_frac.sum() / total_volume.sum()
        else:
            avg_roy_fraction = 0.0
    net_royalty_total = net_rev_total * avg_roy_fraction
    # Discounted cash flow
    discounted_cf = net_royalty_total * discount_factors
    # Cumulative cash flow
    cumulative_cf = np.cumsum(net_royalty_total)
    cumulative_discounted = np.cumsum(discounted_cf)
    # Determine breakeven month (first month where cumulative cash flow turns positive)
    payback_idx = np.where(cumulative_cf > 0)[0]
    payback_date = price_deck['date'].iloc[payback_idx[0]] if len(payback_idx) > 0 else None
    # Build DataFrame of results
    result = pd.DataFrame(
        {
            'date': price_deck['date'],
            'gross_revenue': gross_rev_total,
            'taxes': tax_total + ad_val_total,
            'post_prod_costs': post_prod_total,
            'net_revenue': net_rev_total,
            'net_royalty': net_royalty_total,
            'discount_factor': discount_factors,
            'discounted_cf': discounted_cf,
            'cumulative_cf': cumulative_cf,
            'cumulative_discounted': cumulative_discounted,
        }
    )
    # Append summary metrics as attributes
    result.attrs['undiscounted_total'] = net_royalty_total.sum()
    result.attrs['discounted_total'] = discounted_cf.sum()
    result.attrs['payback_date'] = payback_date
    result.attrs['pv_by_type'] = pv_by_type
    result.attrs['undiscounted_by_type'] = undiscounted_by_type
    return result


def run_monte_carlo(
    wells: List[Well],
    price_deck: pd.DataFrame,
    severance_tax_oil: float,
    severance_tax_gas: float,
    ad_valorem_tax: float,
    post_prod_cost_pct: float,
    discount_rate: float,
    forecast_months: int,
    model_start: datetime.date,
    iterations: int = 100,
    price_sigma: float = 0.10,
    vol_sigma: float = 0.10,
) -> np.ndarray:
    """
    Perform a simple Monte Carlo simulation on price and production assumptions.

    Each iteration perturbs the entire price deck by a normally distributed
    multiplier (mean 1, standard deviation `price_sigma`) and perturbs the
    initial production rates of every well by a multiplier (mean 1,
    standard deviation `vol_sigma`).  The decline parameters and start dates
    remain unchanged.

    Returns
    -------
    np.ndarray
        Array of discounted net present values (one per iteration).
    """
    results = []
    # Pre-generate random multipliers for efficiency
    price_factors = np.random.normal(loc=1.0, scale=price_sigma, size=iterations)
    vol_factors = np.random.normal(loc=1.0, scale=vol_sigma, size=iterations)
    for i in range(iterations):
        # Adjust price deck
        deck = price_deck.copy()
        deck['oil'] *= price_factors[i]
        deck['gas'] *= price_factors[i]
        deck['ngl'] *= price_factors[i]
        # Adjust wells
        sim_wells: List[Well] = []
        for well in wells:
            new_well = Well(
                name=well.name,
                well_type=well.well_type,
                first_prod_date=well.first_prod_date,
                qi_oil=well.qi_oil * vol_factors[i],
                qi_gas=well.qi_gas * vol_factors[i],
                qi_ngl=well.qi_ngl * vol_factors[i],
                b_factor=well.b_factor,
                initial_decline=well.initial_decline,
                terminal_decline=well.terminal_decline,
                royalty_decimal=well.royalty_decimal,
                nri=well.nri,
            )
            new_well.set_start_index(model_start)
            sim_wells.append(new_well)
        sim_result = compute_cash_flows(
            wells=sim_wells,
            price_deck=deck,
            severance_tax_oil=severance_tax_oil,
            severance_tax_gas=severance_tax_gas,
            ad_valorem_tax=ad_valorem_tax,
            post_prod_cost_pct=post_prod_cost_pct,
            discount_rate=discount_rate,
            forecast_months=forecast_months,
        )
        results.append(sim_result.attrs['discounted_total'])
    return np.array(results)


# -----------------------------------------------------------------------------
# Decline Curve Analysis utilities
# -----------------------------------------------------------------------------
def exp_decline(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Exponential decline model: q = qi * exp(-Di * t)."""
    return qi * np.exp(-Di * t)


def hyp_decline(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """Hyperbolic decline model: q = qi / (1 + b * Di * t)**(1/b)."""
    return qi / np.power(1 + b * Di * t, 1.0 / b)


def harm_decline(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Harmonic decline model: q = qi / (1 + Di * t)."""
    return qi / (1 + Di * t)


def fit_decline_model(
    t: np.ndarray,
    q: np.ndarray,
    model: str,
    b_override: Optional[float] = None,
    min_terminal_decline: float = 0.05,
) -> dict:
    """
    Fit a decline curve model to production data using non-linear regression.

    Parameters
    ----------
    t : np.ndarray
        Time in months since first production.
    q : np.ndarray
        Production rate per month (units per month or per day; relative scale is preserved).
    model : str
        Decline model type: 'Exponential', 'Hyperbolic', or 'Harmonic'.
    b_override : Optional[float]
        If provided and the model is hyperbolic, force the b‑factor to this value instead of fitting.
    min_terminal_decline : float
        Minimum terminal decline rate per year used to compute Dₜ in exponential tail.  Not used directly in fitting but returned for reference.

    Returns
    -------
    dict
        Dictionary containing fitted parameters (qi, Di, b), model type, R², RMSE, and predicted values.
    """
    # Provide reasonable initial guesses
    qi0 = max(q[0], 1e-6)
    # Estimate initial decline from the first two points using log difference
    if len(q) > 1 and q[1] > 0 and q[0] > 0:
        Di0 = max((np.log(q[0]) - np.log(q[1])) / (t[1] - t[0]), 1e-6)
    else:
        Di0 = 0.1
    try:
        if model == 'Exponential':
            popt, _ = curve_fit(exp_decline, t, q, p0=[qi0, Di0], bounds=(0, [np.inf, 1]))
            qi_fit, Di_fit = popt
            b_fit = 0.0
            q_pred = exp_decline(t, qi_fit, Di_fit)
        elif model == 'Harmonic':
            popt, _ = curve_fit(harm_decline, t, q, p0=[qi0, Di0], bounds=(0, [np.inf, 1]))
            qi_fit, Di_fit = popt
            b_fit = 1.0
            q_pred = harm_decline(t, qi_fit, Di_fit)
        elif model == 'Hyperbolic':
            if b_override is not None:
                # Fit only qi and Di with fixed b
                def hyp_fixed(t, qi, Di):
                    return hyp_decline(t, qi, Di, b_override)
                popt, _ = curve_fit(hyp_fixed, t, q, p0=[qi0, Di0], bounds=(0, [np.inf, 1]))
                qi_fit, Di_fit = popt
                b_fit = b_override
                q_pred = hyp_decline(t, qi_fit, Di_fit, b_fit)
            else:
                popt, _ = curve_fit(hyp_decline, t, q, p0=[qi0, Di0, 0.5], bounds=(0, [np.inf, 1, 2]))
                qi_fit, Di_fit, b_fit = popt
                q_pred = hyp_decline(t, qi_fit, Di_fit, b_fit)
        else:
            raise ValueError('Unknown decline model')
    except Exception as e:
        return {
            'model': model,
            'qi': np.nan,
            'Di': np.nan,
            'b': np.nan,
            'error': str(e),
        }
    # Goodness-of-fit metrics
    ss_res = np.sum((q - q_pred) ** 2)
    ss_tot = np.sum((q - np.mean(q)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(ss_res / len(q))
    return {
        'model': model,
        'qi': qi_fit,
        'Di': Di_fit,
        'b': b_fit,
        'R2': r_squared,
        'RMSE': rmse,
        'q_pred': q_pred,
        'min_terminal_decline': min_terminal_decline,
    }


# -----------------------------------------------------------------------------
# Date parsing utilities for the Decline Curve Analysis (DCA) module
# -----------------------------------------------------------------------------
def auto_parse_date_series(date_series: pd.Series) -> pd.Series:
    """Attempt to automatically parse a pandas Series of date strings into datetime.

    The function tries several parsing strategies to recognize common
    formats including MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, and month‑year
    strings like Jan-24. It also attempts to interpret numeric values as
    Excel serial date numbers. Any unrecognized values remain as NaT.

    Parameters
    ----------
    date_series : pd.Series
        Series containing raw date values (strings or numbers).

    Returns
    -------
    pd.Series
        Series of datetime64[ns] with the same index as the input.
    """
    series_str = date_series.astype(str).str.strip()
    # Default parsing (year‑first or unambiguous formats)
    parsed = pd.to_datetime(series_str, errors='coerce', infer_datetime_format=True, dayfirst=False)
    invalid = parsed.isna()
    # Try dayfirst parsing for DD/MM/YYYY
    if invalid.any():
        parsed_dayfirst = pd.to_datetime(series_str, errors='coerce', dayfirst=True)
        parsed = parsed.where(~invalid, parsed_dayfirst)
        invalid = parsed.isna()
    # Try month‑year abbreviations (e.g., Jan-24)
    if invalid.any():
        parsed_my_abbrev = pd.to_datetime(series_str, errors='coerce', format='%b-%y')
        parsed = parsed.where(~invalid, parsed_my_abbrev)
        invalid = parsed.isna()
    # Try full month names with year (e.g., January-24)
    if invalid.any():
        parsed_my_full = pd.to_datetime(series_str, errors='coerce', format='%B-%y')
        parsed = parsed.where(~invalid, parsed_my_full)
        invalid = parsed.isna()
    # Attempt Excel serial numbers for numeric strings
    if invalid.any():
        for idx in parsed[invalid].index:
            value = series_str.loc[idx]
            try:
                float_val = float(value)
                if float_val > 59:
                    excel_days = int(float_val)
                    parsed.loc[idx] = pd.to_datetime(excel_days, unit='D', origin='1899-12-30')
            except Exception:
                pass
    return parsed

# -----------------------------------------------------------------------------
# Monte Carlo simulation helper
# -----------------------------------------------------------------------------
def run_monte_carlo_forecast(
    wells_local: List[Well],
    fm_wells: List[FMWell],
    model_start: datetime.date,
    forecast_months: int,
    base_deck: pd.DataFrame,
    diff_oil: float,
    diff_gas: float,
    diff_ngl: float,
    post_prod_pct: float,
    severance_tax_oil: float,
    severance_tax_gas: float,
    discount_rate: float,
    acquisition_cost: float,
    iterations: int,
    price_sigma: float,
    vol_sigma: float,
    ) -> np.ndarray:
    """
    Run a Monte Carlo simulation on price and production assumptions using
    the improved forecasting engine.  Each iteration perturbs the price deck
    and initial rates, then computes the discounted present value (NPV)
    relative to the acquisition cost.

    Parameters
    ----------
    wells_local : List[Well]
        The original well objects defined via the UI, used to preserve well
        types for Monte Carlo runs.
    fm_wells : List[FMWell]
        The corresponding immutable Well objects from the forecast_model
        module.  These provide the base parameters for simulation.
    model_start : datetime.date
        Start date of the forecast.
    forecast_months : int
        Forecast horizon in months.
    base_deck : pd.DataFrame
        Baseline price deck generated from user inputs (without differentials).
    diff_oil, diff_gas, diff_ngl : float
        Price differentials to apply to oil, gas and NGL prices.
    post_prod_pct : float
        Post‑production cost expressed as a fraction of revenue.
    severance_tax_oil, severance_tax_gas : float
        Severance tax rates for oil and gas (NGL uses oil rate).
    discount_rate : float
        Annual discount rate for present value calculations.
    acquisition_cost : float
        Total capital outlay for the deal (lump sum or price × acres).
    iterations : int
        Number of Monte Carlo samples to run.
    price_sigma : float
        Standard deviation of price multipliers (normal distribution).
    vol_sigma : float
        Standard deviation of volume multipliers (normal distribution).

    Returns
    -------
    np.ndarray
        Array of discounted net present values (NPVs) after subtracting
        the acquisition cost.
    """
    rng = np.random.default_rng()
    # Monthly discount factor derived from annual rate
    r_m = (1.0 + discount_rate) ** (1.0 / 12.0) - 1.0
    results: List[float] = []
    for _ in range(int(iterations)):
        # Perturb entire price deck by a single normal multiplier
        price_mult = rng.normal(loc=1.0, scale=price_sigma)
        vol_mult = rng.normal(loc=1.0, scale=vol_sigma)
        # Apply price multiplier to each commodity
        deck = base_deck.copy()
        deck['oil'] = deck['oil'] * price_mult
        deck['gas'] = deck['gas'] * price_mult
        deck['ngl'] = deck['ngl'] * price_mult
        # Aggregate net cash flows across wells
        agg_net = np.zeros(forecast_months)
        for w_local, w in zip(wells_local, fm_wells):
            # Scale initial rates for this iteration
            w_new = FMWell(
                name=w.name,
                first_prod_date=w.first_prod_date,
                qi_oil=w.qi_oil * vol_mult,
                qi_gas=w.qi_gas * vol_mult,
                qi_ngl=w.qi_ngl * vol_mult,
                initial_decline=w.initial_decline,
                b_factor=w.b_factor,
                terminal_decline=w.terminal_decline,
                royalty_decimal=w.royalty_decimal,
                nri=w.nri,
            )
            # Compute cash flows for this perturbed well
            df_cf, _ = compute_cash_flows(
                wells=[w_new],
                start_date=pd.to_datetime(model_start),
                months=forecast_months,
                price_deck=deck,
                royalty_decimal=w_new.royalty_decimal,
                nri=w_new.nri,
                severance_tax_pct_oil=severance_tax_oil,
                severance_tax_pct_gas=severance_tax_gas,
                severance_tax_pct_ngl=severance_tax_oil,
                oil_diff=diff_oil,
                gas_diff=diff_gas,
                ngl_diff=diff_ngl,
                transport_cost=0.0,
                post_prod_cost_pct=post_prod_pct,
                other_fixed_cost_per_month=0.0,
            )
            agg_net += df_cf['net_cash_flow'].to_numpy()
        # Discount cash flows and subtract acquisition cost
        discount_factors = 1.0 / np.power(1.0 + r_m, np.arange(forecast_months))
        discounted_total = float((agg_net * discount_factors).sum())
        results.append(discounted_total - acquisition_cost)
    return np.array(results)


def render_app() -> None:
    """Main function to render the Streamlit application."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Please install streamlit and run\n"
            "`streamlit run dcf_model_web_app.py` from the command line."
        )
    # Sidebar – global assumptions
    st.sidebar.header('Global Assumptions')
    model_start = st.sidebar.date_input(
        'Model start date',
        value=datetime.date(datetime.date.today().year, datetime.date.today().month, 1),
        help='All production and price forecasting begins on this date.'
    )
    forecast_months = st.sidebar.number_input(
        'Forecast horizon (months)',
        min_value=12,
        max_value=480,
        value=360,
        step=12,
        help='Number of months to forecast (e.g., 360 for 30 years).'
    )
    # Discount rate selection
    disc_rate_option = st.sidebar.selectbox(
        'Discount rate option',
        ['PV10 (10 %)', 'PV15 (15 %)', 'Custom'],
    )
    if disc_rate_option == 'PV10 (10 %)':
        discount_rate = 0.10
    elif disc_rate_option == 'PV15 (15 %)':
        discount_rate = 0.15
    else:
        discount_rate = st.sidebar.number_input(
            'Custom discount rate (annual %)',
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.005,
            help='Enter a decimal (e.g., 0.12 for 12 % annual rate).'
        )
    st.sidebar.markdown('---')
    # Pricing assumptions
    st.sidebar.subheader('Pricing')
    price_option = st.sidebar.selectbox(
        'Price deck option', ['Flat pricing', 'Upload market strip'],
        help='Choose a flat price deck or upload your own price forecasts.'
    )
    oil_price = st.sidebar.number_input(
        'Oil price ($/bbl)', value=70.0, min_value=0.0, help='Flat oil price if using flat pricing.'
    )
    gas_price = st.sidebar.number_input(
        'Gas price ($/MMBtu)', value=3.50, min_value=0.0, help='Flat gas price if using flat pricing.'
    )
    ngl_price = st.sidebar.number_input(
        'NGL price ($/bbl)', value=25.0, min_value=0.0, help='Flat NGL price if using flat pricing.'
    )
    # Differentials
    diff_oil = st.sidebar.number_input('Oil price differential ($/bbl)', value=3.0)
    diff_gas = st.sidebar.number_input('Gas price differential ($/MMBtu)', value=0.50)
    diff_ngl = st.sidebar.number_input('NGL price differential ($/bbl)', value=2.0)
    # Taxes
    st.sidebar.subheader('Taxes & Costs')
    severance_tax_oil = st.sidebar.number_input(
        'Oil severance tax (%)', value=4.6, min_value=0.0, max_value=100.0, step=0.1,
        help='Oil & NGL production tax; default 4.6 % of market value【873808517576874†L310-L323】.'
    ) / 100.0
    severance_tax_gas = st.sidebar.number_input(
        'Gas severance tax (%)', value=7.5, min_value=0.0, max_value=100.0, step=0.1,
        help='Natural gas production tax; default 7.5 % of gross value【951814183832884†L85-L88】.'
    ) / 100.0
    ad_valorem_tax = st.sidebar.number_input(
        'Ad valorem tax (%)', value=1.0, min_value=0.0, max_value=100.0, step=0.1,
        help='County property tax on mineral interests; varies by county【752494034296951†L475-L491】.'
    ) / 100.0
    post_prod_pct = st.sidebar.number_input(
        'Post‑production cost (%)', value=0.0, min_value=0.0, max_value=50.0, step=0.1,
        help='Enter a percentage of revenue deducted for post‑production (gathering, compression, transportation).' 
    ) / 100.0

    # Deal economics: allow the user to specify an acquisition cost either as a
    # lump sum or as price per net royalty acre (NRA) times acres.  This
    # capital outlay will be used to compute IRR, payback periods, MOIC and
    # profitability index in the DCF summary.
    st.sidebar.markdown('---')
    st.sidebar.subheader('Deal Economics')
    capital_method = st.sidebar.selectbox(
        'Acquisition cost input',
        ['None', 'Lump sum', 'Price per NRA'],
        help='Choose how to specify the purchase price for the royalty interest.'
    )
    acquisition_cost: float = 0.0
    if capital_method == 'Lump sum':
        acquisition_cost = st.sidebar.number_input(
            'Acquisition price (USD)', value=0.0, min_value=0.0, step=1000.0,
            help='Total lump-sum price paid for the royalty interest.'
        )
    elif capital_method == 'Price per NRA':
        price_per_nra = st.sidebar.number_input(
            'Price per net royalty acre ($/NRA)', value=0.0, min_value=0.0, step=10.0,
            help='Price paid per net royalty acre.'
        )
        nra = st.sidebar.number_input(
            'Net royalty acres (NRA)', value=0.0, min_value=0.0, step=1.0,
            help='Number of net royalty acres purchased.'
        )
        acquisition_cost = price_per_nra * nra
    st.sidebar.markdown('---')
    # Upload price deck if selected
    uploaded_deck = None
    if price_option == 'Upload market strip':
        uploaded_file = st.sidebar.file_uploader(
            'Upload CSV with columns: date, oil, gas, ngl',
            type=['csv'],
            help='The date column should be in YYYY‑MM‑DD format; prices will be net of differentials below.'
        )
        if uploaded_file is not None:
            uploaded_deck = pd.read_csv(uploaded_file)
    # Build the base price deck (without differentials).  Differentials are
    # applied later in the cash flow computation.  We convert the model start
    # date to a pandas Timestamp to satisfy the forecast_model API.
    price_deck = fm_build_price_deck(
        start_date=pd.to_datetime(model_start),
        months=forecast_months,
        oil_start=oil_price,
        oil_mom_growth=0.0,
        gas_start=gas_price,
        gas_mom_growth=0.0,
        ngl_start=ngl_price,
        ngl_mom_growth=0.0,
        custom_df=uploaded_deck,
    )
    # Page layout using tabs to separate DCF model from decline curve analysis
    st.title('Oil & Gas Royalty Toolkit')
    st.write(
        "Use the tabs below to build a discounted cash flow (DCF) forecast or to\n"
        "perform decline curve analysis (DCA) on historical production data."
    )
    # Use three tabs: DCF model, decline curve analysis, and a help/tutorial
    tab_dcf, tab_dca, tab_help = st.tabs(['DCF Model', 'Decline Curve Analysis', 'Help & Tutorial'])
    # -------------------------------
    # DCF Model Tab
    # -------------------------------
    with tab_dcf:
        st.header('DCF Model')
        # Well inputs
        well_count = st.number_input(
            'Number of wells to model', min_value=1, max_value=15, value=3, step=1
        )
        wells: List[Well] = []
        # Manual well definition
        for i in range(int(well_count)):
            with st.expander(f'Well {i+1} inputs'):
                name = st.text_input('Well name/ID', value=f'Well {i+1}', key=f'name_{i}')
                well_type = st.selectbox(
                    'Well type', options=['PDP', 'PUD', 'Future'], key=f'type_{i}'
                )
                first_prod = st.date_input(
                    'First production date',
                    value=model_start + datetime.timedelta(days=30 * i),
                    min_value=model_start,
                    help='For PUD or Future wells, set a future date when production will begin.',
                    key=f'fpd_{i}'
                )
                qi_oil = st.number_input(
                    'Initial oil rate (bbl/d)', value=300.0, min_value=0.0, step=50.0, key=f'qi_oil_{i}'
                )
                qi_gas = st.number_input(
                    'Initial gas rate (Mcf/d)', value=800.0, min_value=0.0, step=100.0, key=f'qi_gas_{i}'
                )
                qi_ngl = st.number_input(
                    'Initial NGL rate (bbl/d)', value=20.0, min_value=0.0, step=5.0, key=f'qi_ngl_{i}'
                )
                b_factor = st.number_input(
                    'Arps b‑factor', value=0.5, min_value=0.0, max_value=2.0, step=0.1, key=f'b_{i}',
                    help='Typical shale wells have b between 0.3 and 0.8【787064351170111†L75-L87】.'
                )
                initial_decline = st.number_input(
                    'Initial decline rate (%)', value=70.0, min_value=0.0, max_value=100.0, step=1.0, key=f'di_{i}',
                    help='Decline in the first year; shale wells often decline 64–70 %【300152174197150†L225-L235】.'
                ) / 100.0
                terminal_decline = st.number_input(
                    'Terminal decline rate (%)', value=5.0, min_value=0.0, max_value=30.0, step=0.5, key=f'dt_{i}',
                    help='Long‑term exponential decline rate after the hyperbolic period.'
                ) / 100.0
                royalty_decimal = st.number_input(
                    'Royalty decimal (fraction)', value=0.1875, min_value=0.0, max_value=1.0, step=0.001, key=f'roy_{i}',
                    help='Gross royalty interest owned (e.g., 0.1875 for 3/16ths).' 
                )
                nri = st.number_input(
                    'Net revenue interest (NRI)', value=0.80, min_value=0.0, max_value=1.0, step=0.01, key=f'nri_{i}',
                    help='Your share after burdens such as overriding royalty interests.'
                )
                # Create Well object
                well = Well(
                    name=name,
                    well_type=well_type,
                    first_prod_date=first_prod,
                    qi_oil=qi_oil,
                    qi_gas=qi_gas,
                    qi_ngl=qi_ngl,
                    b_factor=b_factor,
                    initial_decline=initial_decline,
                    terminal_decline=terminal_decline,
                    royalty_decimal=royalty_decimal,
                    nri=nri,
                )
                wells.append(well)
        # Auto-generate future wells based on acreage and spacing
        with st.expander('Auto‑generate future wells based on acreage'):
            st.write(
                'Use this tool to create a series of future wells based on your available acreage and spacing assumptions.\n'
                'For example, 660‑ft spacing approximates 40 acres per location, while 880‑ft spacing approximates 80 acres.'
            )
            acreage = st.number_input('Tract size (acres)', value=0.0, min_value=0.0, step=10.0)
            spacing_choice = st.selectbox('Spacing', ['660 ft (~40 acres)', '880 ft (~80 acres)'])
            area_per_well = 40.0 if spacing_choice.startswith('660') else 80.0
            first_future = st.date_input('First future well first production date', value=model_start)
            months_between = st.number_input('Months between future wells', value=12, min_value=1, step=1)
            # Default type curve for future wells
            fut_qi_oil = st.number_input('Future well initial oil rate (bbl/d)', value=250.0)
            fut_qi_gas = st.number_input('Future well initial gas rate (Mcf/d)', value=700.0)
            fut_qi_ngl = st.number_input('Future well initial NGL rate (bbl/d)', value=15.0)
            fut_b = st.number_input('Future well b‑factor', value=0.5)
            fut_initial_decline = st.number_input('Future well initial decline rate (%)', value=70.0) / 100.0
            fut_terminal_decline = st.number_input('Future well terminal decline rate (%)', value=5.0) / 100.0
            fut_royalty_decimal = st.number_input('Future well royalty decimal', value=0.1875)
            fut_nri = st.number_input('Future well NRI', value=0.80)
            if st.button('Generate future wells'):
                if acreage > 0:
                    num_locations = int(math.floor(acreage / area_per_well))
                    for j in range(num_locations):
                        prod_date = first_future + datetime.timedelta(days=30 * j * months_between)
                        new_well = Well(
                            name=f'Future {j+1}',
                            well_type='Future',
                            first_prod_date=prod_date,
                            qi_oil=fut_qi_oil,
                            qi_gas=fut_qi_gas,
                            qi_ngl=fut_qi_ngl,
                            b_factor=fut_b,
                            initial_decline=fut_initial_decline,
                            terminal_decline=fut_terminal_decline,
                            royalty_decimal=fut_royalty_decimal,
                            nri=fut_nri,
                        )
                        wells.append(new_well)
                    st.success(f'Added {num_locations} future wells.')
                else:
                    st.warning('Please enter a positive tract size to generate future wells.')
        # Set start month indices for local wells (legacy support).  The improved
        # cash flow engine uses first_prod_date directly, but we preserve
        # start_index computation in case other routines rely on it.
        for well in wells:
            well.set_start_index(model_start)
        # Compute cash flows and investment metrics using the improved engine
    if st.button('Run forecast'):
        # Convert each UI-defined well into the immutable Well used by the
        # forecasting engine.  We also capture the original well type for
        # presenting PV breakdowns by PDP, PUD and Future.
        fm_wells: List[FMWell] = []
        well_types: List[str] = []
        for w in wells:
            fm_wells.append(
                FMWell(
                    name=w.name,
                    first_prod_date=pd.to_datetime(w.first_prod_date),
                    qi_oil=w.qi_oil,
                    qi_gas=w.qi_gas,
                    qi_ngl=w.qi_ngl,
                    initial_decline=w.initial_decline,
                    b_factor=w.b_factor,
                    terminal_decline=w.terminal_decline,
                    royalty_decimal=w.royalty_decimal,
                    nri=w.nri,
                )
            )
            well_types.append(w.well_type)
        # Rebuild the price deck (base) in case the user has changed pricing inputs
        deck = fm_build_price_deck(
            start_date=pd.to_datetime(model_start),
            months=forecast_months,
            oil_start=oil_price,
            oil_mom_growth=0.0,
            gas_start=gas_price,
            gas_mom_growth=0.0,
            ngl_start=ngl_price,
            ngl_mom_growth=0.0,
            custom_df=uploaded_deck,
        )
        # Aggregate net cash flows across all wells and compute PV by type
        agg_net = np.zeros(int(forecast_months))
        pv_by_type: dict[str, float] = {'PDP': 0.0, 'PUD': 0.0, 'Future': 0.0}
        undiscounted_by_type: dict[str, float] = {'PDP': 0.0, 'PUD': 0.0, 'Future': 0.0}
        # Monthly discount factors derived from the annual discount rate
        r_m = (1.0 + discount_rate) ** (1.0 / 12.0) - 1.0
        discount_factors = 1.0 / np.power(1.0 + r_m, np.arange(int(forecast_months)))
        for w_local, w_type, fw in zip(wells, well_types, fm_wells):
            # Compute cash flows for this well using the improved engine
            df_cf, _ = compute_cash_flows(
                wells=[fw],
                start_date=pd.to_datetime(model_start),
                months=int(forecast_months),
                price_deck=deck,
                royalty_decimal=fw.royalty_decimal,
                nri=fw.nri,
                severance_tax_pct_oil=severance_tax_oil,
                severance_tax_pct_gas=severance_tax_gas,
                severance_tax_pct_ngl=severance_tax_oil,
                oil_diff=diff_oil,
                gas_diff=diff_gas,
                ngl_diff=diff_ngl,
                transport_cost=0.0,
                post_prod_cost_pct=post_prod_pct,
                other_fixed_cost_per_month=0.0,
            )
            net_cf = df_cf['net_cash_flow'].to_numpy()
            agg_net += net_cf
            # Accumulate PV by well type
            pv_by_type[w_type] += float((net_cf * discount_factors).sum())
            undiscounted_by_type[w_type] += float(net_cf.sum())
        # Create aggregated DataFrame
        result_df = pd.DataFrame({
            'date': deck['date'],
            'net_cash_flow': agg_net,
        })
        # Compute investment metrics (IRR, NPV, MOIC, PI, payback periods)
        metrics = compute_investment_metrics(
            monthly_net_cash=agg_net,
            acquisition_price=acquisition_cost,
            discount_rate_annual=discount_rate,
        )
        # Display summary results
        st.subheader('Summary Results')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('NPV ($)', f"{metrics['NPV']:,.0f}")
        col2.metric('IRR', f"{metrics['IRR'] * 100:.1f}%" if metrics['IRR'] is not None else 'N/A')
        col3.metric('MOIC (x)', f"{metrics['MOIC']:.2f}" if metrics['MOIC'] is not None else 'N/A')
        col4.metric('Profitability Index', f"{metrics['PI']:.2f}" if metrics['PI'] is not None else 'N/A')
        # Payback periods
        col1, col2 = st.columns(2)
        pb_months = metrics['PaybackMonths']
        dpb_months = metrics['DiscountedPaybackMonths']
        pb_label = 'Payback Period'
        dpb_label = 'Discounted Payback Period'
        if pb_months is not None:
            years, months = divmod(int(pb_months), 12)
            pb_str = f"{years}y {months}m"
        else:
            pb_str = 'N/A'
        if dpb_months is not None:
            dy, dm = divmod(int(dpb_months), 12)
            dpb_str = f"{dy}y {dm}m"
        else:
            dpb_str = 'N/A'
        col1.metric(pb_label, pb_str)
        col2.metric(dpb_label, dpb_str)
        # Present value by well type
        st.markdown('**Present Value by Well Type**')
        pv_type_df = pd.DataFrame({
            'Well type': list(pv_by_type.keys()),
            'PV (USD)': list(pv_by_type.values()),
            'Undiscounted (USD)': list(undiscounted_by_type.values()),
        })
        st.dataframe(pv_type_df)
        # Plot aggregated cumulative cash flows (discounted and undiscounted)
        cum_cf = np.cumsum(agg_net)
        disc_cf = np.cumsum(agg_net * discount_factors)
        cf_plot_df = pd.DataFrame({
            'date': deck['date'],
            'Cumulative CF': cum_cf,
            'Cumulative Discounted CF': disc_cf,
        })
        st.line_chart(cf_plot_df.set_index('date'))
        # Detailed monthly cash flow
        with st.expander('Detailed Cash Flow Table'):
            st.dataframe(result_df)
        # CSV download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download cash flow data as CSV',
            data=csv,
            file_name='royalty_dcf_results.csv',
            mime='text/csv'
        )
        # Monte Carlo simulation option
        st.markdown('---')
        st.subheader('Risk Analysis (Monte Carlo)')
        if st.checkbox('Run Monte Carlo simulation'):
            iterations = st.number_input(
                'Number of iterations', min_value=10, max_value=2000, value=200, step=10,
                help='Higher iterations improve accuracy but increase computation time.'
            )
            price_sigma = st.number_input(
                'Price variability (standard deviation %)', min_value=0.0, max_value=0.5, value=0.10, step=0.01,
                help='Standard deviation of price multiplier. 10 % means prices vary roughly ±10 %.'
            )
            vol_sigma = st.number_input(
                'Production variability (standard deviation %)', min_value=0.0, max_value=0.5, value=0.10, step=0.01,
                help='Standard deviation of production multiplier.'
            )
            if st.button('Run Monte Carlo'):
                with st.spinner('Running simulations...'):
                    mc_results = run_monte_carlo_forecast(
                        wells_local=wells,
                        fm_wells=fm_wells,
                        model_start=model_start,
                        forecast_months=int(forecast_months),
                        base_deck=deck,
                        diff_oil=diff_oil,
                        diff_gas=diff_gas,
                        diff_ngl=diff_ngl,
                        post_prod_pct=post_prod_pct,
                        severance_tax_oil=severance_tax_oil,
                        severance_tax_gas=severance_tax_gas,
                        discount_rate=discount_rate,
                        acquisition_cost=acquisition_cost,
                        iterations=int(iterations),
                        price_sigma=float(price_sigma),
                        vol_sigma=float(vol_sigma),
                    )
                # Percentiles of the NPV distribution
                p10 = np.percentile(mc_results, 10)
                p50 = np.percentile(mc_results, 50)
                p90 = np.percentile(mc_results, 90)
                st.write(
                    f'**P10 NPV:** {p10:,.0f} USD\n\n'
                    f'**P50 (median) NPV:** {p50:,.0f} USD\n\n'
                    f'**P90 NPV:** {p90:,.0f} USD\n\n'
                    'These values represent the distribution of discounted cash flows minus acquisition cost across the simulated scenarios.'
                )
                # Histogram of Monte Carlo results
                hist_df = pd.DataFrame({'NPV': mc_results})
                st.bar_chart(hist_df['NPV'].value_counts().sort_index())
        # Guidance on updating price decks
        st.markdown('---')
        st.markdown(
            '### Updating the Price Deck\n'
            'To update the price deck, select **Upload market strip** under the pricing options in the sidebar.\n'
            'Provide a CSV with the columns `date`, `oil`, `gas`, and `ngl`.  The app will align your prices to the model start date and extend the last available price forward if necessary.\n'
            'If no file is provided, flat prices entered in the sidebar will be used.'
        )
    # -------------------------------
    # Decline Curve Analysis Tab
    # -------------------------------
    with tab_dca:
        st.header('Decline Curve Analysis (DCA)')
        st.write(
            'Upload historical monthly production data to estimate decline parameters.\n'
            'Supported models: exponential, hyperbolic, and harmonic.  The analysis fits a curve to your data and reports qᵢ, Dᵢ and b (if applicable), along with goodness‑of‑fit metrics.'
        )
        prod_file = st.file_uploader('Upload production data (CSV)', type=['csv'])
        if prod_file is not None:
            # New DCA logic begins here
            # Read the uploaded CSV
            try:
                df = pd.read_csv(prod_file)
            except Exception as ex:
                st.error(f"Error reading CSV: {ex}")
                st.stop()
            # Choose the date column
            date_candidates = [col for col in df.columns if "date" in col.lower()]
            if not date_candidates:
                date_candidates = df.columns.tolist()
            date_col = st.selectbox(
                "Select the column containing dates",
                options=date_candidates,
                index=0
            )
            # Toggle for automatic date conversion
            auto_convert = st.checkbox(
                "Auto-convert date formats",
                value=True,
                help="Attempt to automatically recognize common date formats such as MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, month-year (Jan-24) or Excel serial numbers."
            )
            # Parse dates accordingly
            if auto_convert:
                parsed_dates = auto_parse_date_series(df[date_col])
                unresolved = parsed_dates.isna()
                if unresolved.any():
                    st.warning("Some dates could not be recognized automatically. Please select the correct format below.")
                    date_format_options = {
                        "MM/DD/YYYY": "%m/%d/%Y",
                        "DD/MM/YYYY": "%d/%m/%Y",
                        "YYYY-MM-DD": "%Y-%m-%d",
                        "Mon-YY (e.g., Jan-24)": "%b-%y",
                        "Month-YY (e.g., January-24)": "%B-%y",
                        "Excel serial number": "excel"
                    }
                    manual_format = st.selectbox("Select date format for unresolved entries", list(date_format_options.keys()))
                    parsed_manual = parsed_dates.copy()
                    if date_format_options[manual_format] == "excel":
                        for idx in parsed_manual[parsed_manual.isna()].index:
                            try:
                                value = float(df.loc[idx, date_col])
                                if value > 59:
                                    parsed_manual.loc[idx] = pd.to_datetime(int(value), unit="D", origin="1899-12-30")
                            except Exception:
                                pass
                    else:
                        fmt = date_format_options[manual_format]
                        dayfirst = True if fmt.startswith("%d") else False
                        try:
                            parsed_manual = pd.to_datetime(df[date_col], errors="coerce", format=fmt, dayfirst=dayfirst)
                        except Exception:
                            parsed_manual = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
                    parsed_dates = parsed_dates.where(~unresolved, parsed_manual)
            else:
                st.info("Automatic date conversion disabled. Please choose the date format below:")
                date_format_options = {
                    "MM/DD/YYYY": "%m/%d/%Y",
                    "DD/MM/YYYY": "%d/%m/%Y",
                    "YYYY-MM-DD": "%Y-%m-%d",
                    "Mon-YY (e.g., Jan-24)": "%b-%y",
                    "Month-YY (e.g., January-24)": "%B-%y",
                    "Excel serial number": "excel"
                }
                manual_format = st.selectbox("Select date format", list(date_format_options.keys()))
                if date_format_options[manual_format] == "excel":
                    parsed_dates = pd.to_datetime(pd.to_numeric(df[date_col], errors="coerce"), unit="D", origin="1899-12-30")
                else:
                    fmt = date_format_options[manual_format]
                    dayfirst = True if fmt.startswith("%d") else False
                    parsed_dates = pd.to_datetime(df[date_col], errors="coerce", format=fmt, dayfirst=dayfirst)
            # Preview parsed dates
            preview_df = pd.DataFrame({
                "Original": df[date_col].astype(str).head(10),
                "Parsed": parsed_dates.dt.strftime("%Y-%m-%d").head(10)
            })
            st.markdown("**Date preview (first 10 rows):**")
            st.dataframe(preview_df)
            # Validate parsed dates
            if parsed_dates.isna().all():
                st.error("All dates are invalid. Please check your date column and format.")
                st.stop()
            elif parsed_dates.isna().any():
                st.warning("Some dates could not be parsed and will be ignored.")
            df["date_parsed"] = parsed_dates
            df = df.dropna(subset=["date_parsed"]).copy()

            # Automatically detect a well identifier column. Candidates include any column (besides the date)
            # whose name contains keywords such as 'well', 'api', 'id' or 'name'.  If none are found,
            # assume all rows belong to a single well.
            potential_well_cols = []
            for col in df.columns:
                if col == date_col:
                    continue
                c_lower = col.lower()
                if any(k in c_lower for k in ["well", "api", "id", "name"]):
                    potential_well_cols.append(col)
            if potential_well_cols:
                well_id_col = st.selectbox(
                    "Select the column containing well identifiers",
                    options=potential_well_cols,
                    index=0,
                )
            else:
                df["well_id"] = "Well 1"
                well_id_col = "well_id"

            # Determine candidate rate/production columns by inspecting numeric columns and fuzzy matching
            numeric_candidates = []
            for col in df.columns:
                if col in {date_col, well_id_col, "date_parsed"}:
                    continue
                # Check if column contains numeric values
                if pd.to_numeric(df[col], errors="coerce").notna().any():
                    numeric_candidates.append(col)
            # Map columns to stream categories based on fuzzy keywords
            stream_map: dict[str, List[str]] = {}
            for col in numeric_candidates:
                c_lower = col.lower()
                assigned = False
                # Oil
                for kw in ["oil", "oil production", "oil_prod", "oil_rate", "bbl", "bopd"]:
                    if kw in c_lower:
                        stream_map.setdefault("oil", []).append(col)
                        assigned = True
                        break
                if assigned:
                    continue
                # Gas
                for kw in ["gas", "gas production", "gas_prod", "gas_rate", "mcf", "mcfd", "mmcf"]:
                    if kw in c_lower:
                        stream_map.setdefault("gas", []).append(col)
                        assigned = True
                        break
                if assigned:
                    continue
                # NGL / liquids
                for kw in ["ngl", "condensate"]:
                    if kw in c_lower:
                        stream_map.setdefault("ngl", []).append(col)
                        assigned = True
                        break
                if assigned:
                    continue
                # BOE or generic rate/production
                generic_keywords = ["boe", "rate", "prod", "production", "volume", "qty"]
                if any(kw in c_lower for kw in generic_keywords):
                    stream_map.setdefault("rate", []).append(col)
                    assigned = True
                if not assigned:
                    # If it's numeric but unlabelled, include as generic rate
                    stream_map.setdefault("rate", []).append(col)

            # Flatten stream_map to a unique list of rate columns
            rate_columns = sorted(set(sum(stream_map.values(), [])))
            if not rate_columns:
                st.error(
                    "No numeric rate columns were detected. Please ensure your CSV contains "
                    "columns with production volumes or rates."
                )
                st.stop()

            # Provide option to aggregate daily data to monthly totals
            aggregate_monthly = st.checkbox(
                "Aggregate daily data to monthly totals",
                value=False,
                help="If selected, daily production will be summed into monthly totals using the first day of the month.",
            )
            df_work = df.copy()
            if aggregate_monthly:
                df_work["period"] = df_work["date_parsed"].dt.to_period("M")
                # Sum all detected rate columns by well and month
                grouped = df_work.groupby([well_id_col, "period"])[rate_columns].sum().reset_index()
                grouped["date_parsed"] = grouped["period"].dt.to_timestamp()
                grouped.drop(columns=["period"], inplace=True)
                df_work = grouped

            # Sort and compute time since first production for each well
            df_work = df_work.sort_values([well_id_col, "date_parsed"]).copy()
            df_work["t_months"] = df_work.groupby(well_id_col)["date_parsed"].transform(
                lambda s: (s - s.min()).dt.days / 30.4375
            )

            # Ask user which well and which streams to analyse
            unique_wells = df_work[well_id_col].astype(str).unique()
            selected_well = st.selectbox("Select well for analysis", unique_wells)
            selected_streams = st.multiselect(
                "Select the streams you want to analyse",
                options=rate_columns,
                default=rate_columns,
            )
            well_data = df_work[df_work[well_id_col].astype(str) == str(selected_well)].copy()
            if well_data.empty:
                st.warning("No data found for the selected well.")
                st.stop()

            # Fit decline curves for each selected stream
            for stream in selected_streams:
                q_series = pd.to_numeric(well_data[stream], errors="coerce")
                if q_series.isna().all():
                    st.warning(f"No numeric data found for {stream}.")
                    continue
                # Optional outlier removal using 3σ rule
                if st.checkbox(f"Remove outliers for {stream}", value=True):
                    mean_val = q_series.mean()
                    std_val = q_series.std() if q_series.std() > 0 else 1.0
                    mask_vals = (np.abs(q_series - mean_val) <= 3 * std_val)
                    wds = well_data[mask_vals]
                    q_vals = pd.to_numeric(wds[stream], errors="coerce").to_numpy()
                    t_vals = wds["t_months"].to_numpy()
                else:
                    q_vals = pd.to_numeric(well_data[stream], errors="coerce").to_numpy()
                    t_vals = well_data["t_months"].to_numpy()
                # Ensure positive rates
                mask_pos = q_vals > 0
                q_vals = q_vals[mask_pos]
                t_vals = t_vals[mask_pos]
                # Select decline model
                model_choice = st.selectbox(
                    f"Decline model for {stream}",
                    ["Exponential", "Hyperbolic", "Harmonic"],
                )
                b_override = None
                if model_choice == "Hyperbolic" and st.checkbox(
                    f"Manual b‑factor override for {stream}", value=False
                ):
                    b_override = st.number_input(
                        f"b‑factor (override) for {stream}",
                        value=0.5,
                        min_value=0.0,
                        max_value=2.0,
                        step=0.1,
                    )
                min_dt = (
                    st.number_input(
                        f"Minimum terminal decline (%) for {stream}",
                        value=5.0,
                        min_value=0.0,
                        max_value=30.0,
                        step=0.5,
                    )
                    / 100.0
                )
                if len(q_vals) >= 3:
                    result = fit_decline_model(
                        t_vals,
                        q_vals,
                        model_choice,
                        b_override=b_override,
                        min_terminal_decline=min_dt,
                    )
                    if "error" in result:
                        st.error(f"Error fitting curve for {stream}: {result['error']}")
                    else:
                        clean_name = stream.replace("_rate", "")
                        st.markdown(f"#### {clean_name.title()} decline fit")
                        st.write(f"Initial rate (qᵢ): {result['qi']:.2f}")
                        st.write(f"Initial decline (Dᵢ): {result['Di']:.4f} per month")
                        st.write(f"b‑factor: {result['b']:.3f}")
                        st.write(f"R²: {result['R2']:.3f}")
                        st.write(f"RMSE: {result['RMSE']:.3f}")
                        fig, ax = plt.subplots()
                        ax.plot(t_vals, q_vals, "o", label=f"Observed {clean_name}")
                        ax.plot(t_vals, result["q_pred"], "-", label=f"Fitted {clean_name}")
                        ax.set_xlabel("Time (months)")
                        ax.set_ylabel(f"Production rate ({clean_name.upper()})")
                        ax.set_title(f"Decline Curve Fit – {clean_name.title()} ({model_choice})")
                        ax.legend()
                        st.pyplot(fig)
                else:
                    st.warning(f"Not enough data points to fit a decline curve for {stream}")
            st.stop()
            if False:
                """
                # In the DCA tab, after reading the uploaded CSV file into a DataFrame `df`:
                df = pd.read_csv(uploaded_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Compute time since first production in months for each well
                df.sort_values(['well_id', 'date'], inplace=True)
                df['t_months'] = (
                    df.groupby('well_id')['date'].transform(lambda s: (s - s.min()).dt.days / 30.4375)
                )
                
                # Identify all columns that look like rate columns.  By convention we assume they end with '_rate'
                rate_columns = [col for col in df.columns if col.endswith('_rate')]
                if not rate_columns:
                    # Fall back to a single generic 'rate' column if present
                    if 'rate' in df.columns:
                        rate_columns = ['rate']
                    else:
                        st.error('No rate columns found in the uploaded CSV. Please include columns like oil_rate or gas_rate.')
                        st.stop()
                
                # Ask the user which well and which commodity streams to analyse
                unique_wells = df['well_id'].unique()
                selected_well = st.selectbox('Select well for analysis', unique_wells)
                available_streams = rate_columns
                selected_streams = st.multiselect(
                    'Select the streams you want to analyse',
                    options=available_streams,
                    default=available_streams  # default to all streams
                )
                
                # Filter the DataFrame for the chosen well
                well_data = df[df['well_id'] == selected_well].copy()
                if well_data.empty:
                    st.warning('No data found for the selected well.')
                    st.stop()
                
                # For each selected stream, fit a decline curve and display results
                for stream in selected_streams:
                    q_series = well_data[stream].astype(float)
                
                    # Simple 3σ outlier removal (optional)
                    if st.checkbox(f'Remove outliers for {stream}', value=True):
                        mean = q_series.mean()
                        std = q_series.std() if q_series.std() > 0 else 1.0
                        mask = (np.abs(q_series - mean) <= 3 * std)
                        well_data_stream = well_data[mask]
                        q_vals = well_data_stream[stream].values
                        t_vals = well_data_stream['t_months'].values
                    else:
                        well_data_stream = well_data
                        q_vals = well_data_stream[stream].values
                        t_vals = well_data_stream['t_months'].values
                
                    # Drop zero or negative rates to avoid issues in curve fitting
                    mask = q_vals > 0
                    q_vals = q_vals[mask]
                    t_vals = t_vals[mask]
                
                    # Choose the decline model and b‑factor override (as before)
                    model_choice = st.selectbox(f'Decline model for {stream}', ['Exponential', 'Hyperbolic', 'Harmonic'])
                    b_override = None
                    if model_choice == 'Hyperbolic' and st.checkbox(f'Manual b‑factor override for {stream}', value=False):
                        b_override = st.number_input(f'b‑factor (override) for {stream}', value=0.5, min_value=0.0, max_value=2.0, step=0.1)
                    min_dt = st.number_input(f'Minimum terminal decline (%) for {stream}', value=5.0, min_value=0.0, max_value=30.0, step=0.5) / 100.0
                
                    # Fit the curve using your existing fit_decline_model() helper
                    if len(q_vals) >= 3:  # need at least 3 points for a meaningful fit
                        result = fit_decline_model(t_vals, q_vals, model_choice, b_override=b_override, min_terminal_decline=min_dt)
                        if 'error' in result:
                            st.error(f"Error fitting curve for {stream}: {result['error']}")
                        else:
                            st.markdown(f'#### {stream.replace('_rate',"").title()} decline fit')
                            st.write(f"Initial rate (qᵢ): {result['qi']:.2f}")
                            st.write(f"Initial decline (Dᵢ): {result['Di']:.4f} per month")
                            st.write(f"b‑factor: {result['b']:.3f}")
                            st.write(f"R²: {result['R2']:.3f}")
                            st.write(f"RMSE: {result['RMSE']:.3f}")
                            # Plot observed vs fitted for this stream
                            fig, ax = plt.subplots()
                            ax.plot(t_vals, q_vals, 'o', label=f'Observed {stream}')
                            ax.plot(t_vals, result['q_pred'], '-', label=f'Fitted {stream}')
                            ax.set_xlabel('Time (months)')
                            # Clean up the stream name for labels/titles
                            base_name = stream.replace('_rate', '')
                            ax.set_ylabel(f'Production rate ({base_name.upper()})')
                            ax.set_title(f'Decline Curve Fit – {base_name.title()} ({model_choice})')
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        st.warning(f'Not enough data points to fit a decline curve for {stream}')
                """
    # -------------------------------------------------------------------------
    # Help & Tutorial Tab
    # -------------------------------------------------------------------------
    with tab_help:
        st.header('Help & Tutorial')
        # Overview section
        st.markdown('### Overview')
        st.markdown(
            'This model evaluates the value of an oil and gas royalty interest using discounted cash flow (DCF) analysis.\n'
            'It forecasts monthly production from your wells, applies price assumptions, taxes and costs, and discounts the resulting cash flows to compute metrics such as net present value (NPV), PV10, PV15 and payback period.\n'
            'The model can handle existing producing wells (PDP), permitted but undrilled wells (PUDs) and future locations, making it suitable for diverse evaluations.\n'
            'For example, you can assess a 1 % royalty in a 640‑acre unit containing four producing wells and two planned drilling spacing units (DSUs).'
        )
        # Workflow diagram
        st.markdown('### Model Workflow')
        # Resolve the image path relative to this file
        flow_path = Path(__file__).with_name('dcf_flow_diagram.png')
        st.image(str(flow_path), caption='High‑level workflow of the DCF model', use_column_width=True)
        # Glossary
        st.markdown('### Glossary of Key Terms')
        glossary_items = [
            ('DCF (Discounted Cash Flow)', 'A valuation method estimating the present value of an investment based on its expected future cash flows【735789832473702†L325-L340】.'),
            ('NPV / PV10 / PV15', 'NPV is the net present value of future cash flows discounted at a chosen rate. PV10 and PV15 are standard discount rates of 10 % and 15 %, commonly used to value reserves【817667812575893†L240-L246】.'),
            ('qᵢ (Initial Production Rate)', 'The initial production rate of a well at the start of the forecast, typically measured in barrels per day or Mcf per day.'),
            ('Dᵢ (Initial Decline Rate)', 'The initial annualized decline rate in the first year of production; shale wells often decline 64–70 %【300152174197150†L225-L235】.'),
            ('b‑factor (Decline Exponent)', 'The Arps b‑factor controls the shape of a decline curve: b=0 gives exponential decline, b=1 harmonic, and 0<b<1 hyperbolic【651949411700584†L112-L117】.'),
            ('NRI (Net Revenue Interest)', 'The proportion of production revenue a royalty owner receives after deducting expenses and royalty burdens【404001467312245†L54-L67】.'),
            ('Royalty Decimal / Gross Royalty', 'The fractional share of production revenue stipulated in a lease, expressed as a decimal (e.g., 0.1875 for 3/16ths).'),
            ('PDP (Proved Developed Producing)', 'Reserves expected to be recovered from currently producing zones under continuation of present operating methods【832034642446589†L18-L22】.'),
            ('PUD (Proved Undeveloped)', 'Proved reserves categorized as undeveloped—hydrocarbon resources requiring new wells or further investment for recovery【449576985926588†L18-L57】.'),
            ('Strip Pricing / Futures Strip', 'A pricing approach using futures contracts in sequential delivery months to lock in commodity prices over a specified time frame【364657486048314†L233-L253】.'),
            ('Price Differential', 'The difference between an established benchmark price and the realized price at the lease or field, accounting for quality, transportation and regional adjustments【483537226548607†L73-L76】.'),
            ('Post‑Production Costs', 'Expenses incurred after extraction—such as transportation, processing, and marketing—that are deducted from revenue before paying royalties【431491580934731†L60-L80】.'),
        ]
        for term, definition in glossary_items:
            st.markdown(f'**{term}**: {definition}')
        # Step‑by‑step guide
        st.markdown('### Step‑by‑Step Guide')
        st.markdown(
            '* **Enter Your Wells** – In the DCF Model tab, specify the number of wells and input well type (PDP, PUD or Future), first production date, initial rates, decline parameters, royalty decimal and NRI for each well.\n'
            '* **Upload or Define Pricing** – Choose Flat pricing or upload a Market Strip from the sidebar. Enter price differentials for oil, gas and NGL.\n'
            '* **Set Global Assumptions** – Specify the model start date, forecast horizon, discount rate, tax rates, and optional post‑production costs.\n'
            '* **Generate Future Wells** – Use the auto‑generate tool to schedule future wells based on acreage and spacing assumptions.\n'
            '* **Run the Forecast** – Click **Run forecast** to compute monthly production, revenue, taxes, costs and net royalty cash flows. The app then calculates NPV, PV10, PV15 and payback period.\n'
            '* **Review Results** – Examine summary metrics, present value by well type, and cumulative cash flow chart. Expand the detailed table or download a CSV.\n'
            '* **Perform Risk Analysis** – Enable the Monte Carlo simulation to see P10/P50/P90 NPVs based on price and volume variability.'
        )
        # DCA guidance
        st.markdown('### Decline Curve Analysis (DCA) Guidance')
        st.markdown(
            'In the DCA tab, upload historical production data (CSV with `date`, `well_id` and `rate`).\n'
            'Choose a decline model – **Exponential**, **Hyperbolic** or **Harmonic** – and optionally override the b‑factor.\n'
            'The model fits a decline curve using non‑linear regression, reports qᵢ, Dᵢ and b (if applicable), and computes R² and RMSE to gauge fit quality.\n'
            'Use the plot of observed versus fitted production rates to validate the fit and adjust parameters as needed.'
        )
        flow_path = Path(__file__).with_name('decline_curve_examples.png')
        st.image(str(flow_path), caption='Example exponential, hyperbolic and harmonic decline curves', use_column_width=True)
        # Downloadable PDF
        flow_path = Path(__file__).with_name('dcf_help_guide.pdf')
        st.markdown('### Downloadable Guide')
        try:
            with open(flow_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
            st.download_button(
                label='Download PDF Guide',
                data=pdf_bytes,
                file_name='DCF_Help_Guide.pdf',
                mime='application/pdf'
            )
        except Exception:
            st.info('PDF guide is not available. Please contact support.')


# If executed as a script via Streamlit, render the app
if __name__ == '__main__':
    if st is not None:
        render_app()
    else:
        print(
            'Streamlit is not installed.  Install streamlit and run this script with:\n'
            '    streamlit run dcf_model_web_app.py'
        )
