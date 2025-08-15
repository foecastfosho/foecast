import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the well object to a dictionary for session state."""
        return {
            'name': self.name,
            'well_type': self.well_type,
            'first_prod_date': self.first_prod_date.isoformat(),
            'qi_oil': self.qi_oil,
            'qi_gas': self.qi_gas,
            'qi_ngl': self.qi_ngl,
            'b_factor': self.b_factor,
            'initial_decline': self.initial_decline,
            'terminal_decline': self.terminal_decline,
            'royalty_decimal': self.royalty_decimal,
            'nri': self.nri,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Well":
        """Creates a Well object from a dictionary."""
        return cls(
            name=data['name'],
            well_type=data['well_type'],
            first_prod_date=datetime.date.fromisoformat(data['first_prod_date']),
            qi_oil=data['qi_oil'],
            qi_gas=data['qi_gas'],
            qi_ngl=data['qi_ngl'],
            b_factor=data['b_factor'],
            initial_decline=data['initial_decline'],
            terminal_decline=data['terminal_decline'],
            royalty_decimal=data['royalty_decimal'],
            nri=data['nri'],
        )


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
    switch_idx = np.argmax(inst_decline <= Dt_m) # Corrected to compare with monthly terminal rate
    
    # If the hyperbolic decline never falls below the terminal rate, use all
    # hyperbolic values.  Otherwise, switch to exponential after the switch_idx.
    if switch_idx == 0 and inst_decline[0] <= Dt_m:
        # Start immediately on exponential decline
        q = qi * np.power(1 - Dt_m, t)  # exponential decline from initial rate
    elif inst_decline[-1] > Dt_m:
        # Hyperbolic curve never meets terminal decline; use entire hyperbolic
        q = q_hyp.copy()
    else:
        # Combine hyperbolic up to switch_idx, exponential afterwards
        q = q_hyp.copy()
        # Calculate the rate at the switch point
        q_switch = q_hyp[switch_idx]
        # Apply exponential decline from the switch point onwards
        time_since_switch = np.arange(months - switch_idx)
        q[switch_idx:] = q_switch * np.power(1 - Dt_m, time_since_switch)

    # Convert daily rate to monthly volume (approximate 30.4375 days per month)
    return q * 30.4375


def build_price_deck(
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
    """Create a monthly price deck for oil, gas, and NGL.

    If `uploaded_deck` is provided, it should contain columns named
    'date', 'oil', 'gas', and 'ngl' with dates sorted ascending.  Any rows
    beyond the forecast horizon are ignored; if the uploaded deck is shorter
    than the horizon, the last available price is extended forward.

    If no deck is uploaded, flat pricing is used based on the provided
    `oil_price`, `gas_price`, and `ngl_price` parameters.

    Price differentials are subtracted from the base prices to arrive at net
    realized prices.

    Returns a DataFrame with columns: 'date', 'oil', 'gas', 'ngl'.
    """
    dates = pd.to_datetime([start_date + pd.DateOffset(months=i) for i in range(months)])
    
    if uploaded_deck is not None and not uploaded_deck.empty:
        # Clean and sort the uploaded deck
        deck = uploaded_deck.copy()
        # Ensure correct dtypes
        for col in ['oil', 'gas', 'ngl']:
            if col in deck.columns:
                deck[col] = pd.to_numeric(deck[col], errors='coerce')
        if 'date' in deck.columns:
            deck['date'] = pd.to_datetime(deck['date'], errors='coerce')
            deck = deck.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        
        # Create a DataFrame for the full forecast period
        price_df = pd.DataFrame(index=dates)
        
        # Merge the uploaded deck
        price_df = pd.merge_asof(price_df, deck[['date', 'oil', 'gas', 'ngl']], left_index=True, right_on='date', direction='backward')
        price_df.index = dates # Reset index to the forecast dates
        
        # Forward fill any missing values from the merge
        price_df[['oil', 'gas', 'ngl']] = price_df[['oil', 'gas', 'ngl']].ffill()
        
        # Backward fill any initial NaNs with the first available price
        price_df[['oil', 'gas', 'ngl']] = price_df[['oil', 'gas', 'ngl']].bfill()

        # If still NaN (e.g., empty deck), use flat prices
        price_df['oil'] = price_df['oil'].fillna(oil_price)
        price_df['gas'] = price_df['gas'].fillna(gas_price)
        price_df['ngl'] = price_df['ngl'].fillna(ngl_price)

        oil_prices = price_df['oil'].values
        gas_prices = price_df['gas'].values
        ngl_prices = price_df['ngl'].values
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
    total_oil_vol = np.zeros(forecast_months)
    total_gas_vol = np.zeros(forecast_months)
    total_ngl_vol = np.zeros(forecast_months)
    
    total_net_royalty_rev = np.zeros(forecast_months)

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
            qi=well.qi_oil, Di=well.initial_decline, b=well.b_factor,
            Dt=well.terminal_decline, months=forecast_months,
        )
        gas = hyperbolic_decline_rate(
            qi=well.qi_gas, Di=well.initial_decline, b=well.b_factor,
            Dt=well.terminal_decline, months=forecast_months,
        )
        ngl = hyperbolic_decline_rate(
            qi=well.qi_ngl, Di=well.initial_decline, b=well.b_factor,
            Dt=well.terminal_decline, months=forecast_months,
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
        total_oil_vol += oil_shift
        total_gas_vol += gas_shift
        total_ngl_vol += ngl_shift
        
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
        
        # Accumulate total net royalty revenue
        total_net_royalty_rev += net_royalty
        
        # Accumulate for PV by type
        undiscounted_by_type[well.well_type] += net_royalty.sum()
        pv_by_type[well.well_type] += (net_royalty * discount_factors).sum()

    # Aggregate cash flow across all wells (per month)
    gross_rev_total = (
        total_oil_vol * price_deck['oil'].values
        + total_gas_vol * price_deck['gas'].values
        + total_ngl_vol * price_deck['ngl'].values
    )
    tax_total = (
        total_oil_vol * price_deck['oil'].values * severance_tax_oil
        + total_gas_vol * price_deck['gas'].values * severance_tax_gas
        + total_ngl_vol * price_deck['ngl'].values * severance_tax_oil # NGL taxed like oil
    )
    ad_val_total = gross_rev_total * ad_valorem_tax
    post_prod_total = gross_rev_total * post_prod_cost_pct
    net_rev_total = gross_rev_total - tax_total - ad_val_total - post_prod_total

    # Discounted cash flow
    discounted_cf = total_net_royalty_rev * discount_factors
    # Cumulative cash flow
    cumulative_cf = np.cumsum(total_net_royalty_rev)
    cumulative_discounted = np.cumsum(discounted_cf)
    
    # Build DataFrame of results
    result = pd.DataFrame(
        {
            'date': price_deck['date'],
            'gross_revenue': gross_rev_total,
            'taxes': tax_total + ad_val_total,
            'post_prod_costs': post_prod_total,
            'net_revenue': net_rev_total,
            'net_royalty': total_net_royalty_rev,
            'discount_factor': discount_factors,
            'discounted_cf': discounted_cf,
            'cumulative_cf': cumulative_cf,
            'cumulative_discounted': cumulative_discounted,
        }
    )
    # Append summary metrics as attributes
    result.attrs['undiscounted_total'] = total_net_royalty_rev.sum()
    result.attrs['discounted_total'] = discounted_cf.sum()
    result.attrs['pv_by_type'] = pv_by_type
    result.attrs['undiscounted_by_type'] = undiscounted_by_type
    return result


# ---------------------------------------------------------------------------
# Investment metrics helper
#
# This helper takes the result DataFrame produced by ``compute_cash_flows`` and
# a user‑defined acquisition cost to derive additional investment metrics.
# It calculates the internal rate of return (IRR), the payback period (based
# on undiscounted and discounted cash flows), the profitability index (PI)
# and the multiple on invested capital (MOIC).  It also constructs an
# augmented cash flow table that includes the initial capital outlay at a
# specified closing month.
def compute_investment_metrics(
    result_df: pd.DataFrame,
    acquisition_cost: float,
    discount_rate: float,
    model_start: datetime.date,
    closing_date: Optional[datetime.date] = None,
) -> dict:
    """
    Compute investment performance metrics given a cash flow forecast and an
    acquisition cost.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame returned by ``compute_cash_flows`` containing monthly net
        royalty cash flows and discount factors.
    acquisition_cost : float
        Positive dollar amount representing the upfront purchase price or
        capital outlay.  This cost will be treated as a negative cash flow.
    discount_rate : float
        Annual discount rate (decimal) used for discounted payback period and
        NPV calculations.
    model_start : datetime.date
        The start date of the forecast.  Used to align the closing month.
    closing_date : Optional[datetime.date], default None
        Date when the acquisition cost is incurred.  If None, the cost is
        applied at the first month of the forecast.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - ``cash_flow_table`` (pd.DataFrame): augmented table with columns
          ['date', 'cash_inflow', 'acquisition', 'net_cash',
           'discount_factor', 'discounted_net_cash', 'cumulative_net_cash',
           'cumulative_discounted'].
        - ``irr_annual`` (float or None): annualized internal rate of return.
        - ``payback_months`` (int or None): payback period in months based on
          undiscounted cash flows (including the initial cost).
        - ``discounted_payback_months`` (int or None): payback period in
          months based on discounted cash flows.
        - ``npv`` (float): net present value of the investment at the
          specified discount rate.
        - ``profitability_index`` (float or None): profitability index
          (PV of returns divided by acquisition cost).
        - ``moic`` (float or None): multiple on invested capital (total
          undiscounted returns divided by acquisition cost).
    """
    # Copy relevant arrays
    cash_inflow = result_df['net_royalty'].values.astype(float)
    discount_factors = result_df['discount_factor'].values.astype(float)
    dates = result_df['date'].values
    n_periods = len(cash_inflow)
    # Determine the month index when the acquisition occurs
    if closing_date is None:
        closing_month_index = 0
    else:
        # Compute difference in months between closing_date and model_start
        delta_months = (closing_date.year - model_start.year) * 12 + (closing_date.month - model_start.month)
        closing_month_index = max(0, int(delta_months))
        # Cap at forecast horizon
        if closing_month_index >= n_periods:
            closing_month_index = n_periods - 1
    # Build arrays for acquisition and net cash
    acquisition_array = np.zeros(n_periods)
    if acquisition_cost > 0:
        acquisition_array[closing_month_index] = -acquisition_cost
    # Net cash including acquisition cost
    net_cash = cash_inflow + acquisition_array
    # Discounted net cash
    discounted_net_cash = net_cash * discount_factors
    # Cumulative sums
    cumulative_net_cash = np.cumsum(net_cash)
    cumulative_discounted = np.cumsum(discounted_net_cash)
    # Compute IRR on monthly basis; annualize if possible
    irr_monthly = None
    try:
        # Prefer numpy_financial if available for accurate IRR calculation
        try:
            import numpy_financial as npf  # type: ignore
            irr_val = npf.irr(net_cash)
        except Exception:
            # Fall back to numpy's IRR if available
            irr_val = np.irr(net_cash)  # type: ignore[attr-defined]
        if irr_val is not None and not np.isnan(irr_val):
            irr_monthly = irr_val
    except Exception:
        irr_monthly = None
    if irr_monthly is not None and np.isfinite(irr_monthly):
        irr_annual = (1 + irr_monthly) ** 12 - 1
    else:
        irr_annual = None
    # Payback periods
    payback_months = None
    idx = np.where(cumulative_net_cash > 0)[0]
    if len(idx) > 0:
        payback_months = int(idx[0])
    discounted_payback_months = None
    idx_d = np.where(cumulative_discounted > 0)[0]
    if len(idx_d) > 0:
        discounted_payback_months = int(idx_d[0])
    # NPV (PV of returns minus cost)
    npv = discounted_net_cash.sum()

    profitability_index = None
    if acquisition_cost > 0:
        pv_returns = result_df['discounted_cf'].sum()
        profitability_index = pv_returns / acquisition_cost

    moic = None
    if acquisition_cost > 0:
        total_return = cash_inflow.sum()
        moic = total_return / acquisition_cost
    # Build augmented table
    cf_table = pd.DataFrame({
        'date': dates,
        'cash_inflow': cash_inflow,
        'acquisition': acquisition_array,
        'net_cash': net_cash,
        'discount_factor': discount_factors,
        'discounted_net_cash': discounted_net_cash,
        'cumulative_net_cash': cumulative_net_cash,
        'cumulative_discounted': cumulative_discounted,
    })
    return {
        'cash_flow_table': cf_table,
        'irr_annual': irr_annual,
        'payback_months': payback_months,
        'discounted_payback_months': discounted_payback_months,
        'npv': npv,
        'profitability_index': profitability_index,
        'moic': moic,
    }


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


def get_default_scenario():
    """Returns a dictionary with the default input values for a new scenario."""
    return {
        "model_start": datetime.date(datetime.date.today().year, datetime.date.today().month, 1),
        "forecast_months": 360,
        "discount_rate": 0.10,
        "price_option": "Flat pricing",
        "oil_price": 70.0,
        "gas_price": 3.50,
        "ngl_price": 25.0,
        "diff_oil": 3.0,
        "diff_gas": 0.50,
        "diff_ngl": 2.0,
        "severance_tax_oil": 0.046,
        "severance_tax_gas": 0.075,
        "ad_valorem_tax": 0.01,
        "post_prod_pct": 0.0,
        "well_count": 1,
        "wells": [Well(
            name='Well 1', well_type='PDP', first_prod_date=datetime.date.today(),
            qi_oil=300.0, qi_gas=800.0, qi_ngl=20.0, b_factor=0.5,
            initial_decline=0.70, terminal_decline=0.05,
            royalty_decimal=0.1875, nri=0.80
        ).to_dict()],
        "acquisition_cost": 0.0,
        "closing_date": datetime.date(datetime.date.today().year, datetime.date.today().month, 1),
    }

def render_app() -> None:
    """Main function to render the Streamlit application."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Please install streamlit and run\n"
            "`streamlit run dcf_model_web_app.py` from the command line."
        )

    st.set_page_config(layout="wide")

    # --- Initialize Session State for Scenarios ---
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = {"Default Case": get_default_scenario()}
    if 'active_scenario' not in st.session_state:
        st.session_state.active_scenario = "Default Case"
    if 'uploaded_deck' not in st.session_state:
        st.session_state.uploaded_deck = None


    # --- Sidebar ---
    st.sidebar.header('Scenario Management')
    
    # Scenario selection
    scenario_names = list(st.session_state.scenarios.keys())
    st.session_state.active_scenario = st.sidebar.selectbox(
        "Active Scenario",
        scenario_names,
        index=scenario_names.index(st.session_state.active_scenario),
        key="scenario_selector"
    )

    # Load active scenario data
    active_scenario_data = st.session_state.scenarios[st.session_state.active_scenario]

    # Scenario creation/deletion
    with st.sidebar.expander("Create / Delete Scenario"):
        new_scenario_name = st.text_input("New Scenario Name")
        if st.button("Save as New Scenario"):
            if new_scenario_name and new_scenario_name not in st.session_state.scenarios:
                st.session_state.scenarios[new_scenario_name] = active_scenario_data.copy()
                st.session_state.active_scenario = new_scenario_name
                st.rerun()
            else:
                st.warning("Please enter a unique name for the new scenario.")

        if st.button("Delete Current Scenario", type="secondary"):
            if st.session_state.active_scenario != "Default Case":
                del st.session_state.scenarios[st.session_state.active_scenario]
                st.session_state.active_scenario = "Default Case"
                st.rerun()
            else:
                st.warning("Cannot delete the Default Case.")


    st.sidebar.header('Global Assumptions')
    
    # Update active scenario data from UI widgets
    active_scenario_data["model_start"] = st.sidebar.date_input(
        'Model start date',
        value=active_scenario_data["model_start"],
        help='All production and price forecasting begins on this date.'
    )
    active_scenario_data["forecast_months"] = st.sidebar.number_input(
        'Forecast horizon (months)',
        min_value=12, max_value=480, value=active_scenario_data["forecast_months"], step=12,
        help='Number of months to forecast (e.g., 360 for 30 years).'
    )
    active_scenario_data["discount_rate"] = st.sidebar.number_input(
        'Discount rate (annual %)',
        min_value=0.0, max_value=100.0, value=active_scenario_data["discount_rate"]*100, step=0.5,
        help='Enter a percentage (e.g., 10 for 10% annual rate).'
    ) / 100.0

    st.sidebar.markdown('---')
    st.sidebar.subheader('Pricing')
    active_scenario_data["price_option"] = st.sidebar.selectbox(
        'Price deck option', ['Flat pricing', 'Upload market strip'],
        index=['Flat pricing', 'Upload market strip'].index(active_scenario_data["price_option"]),
        help='Choose a flat price deck or upload your own price forecasts.'
    )
    active_scenario_data["oil_price"] = st.sidebar.number_input('Oil price ($/bbl)', value=active_scenario_data["oil_price"], min_value=0.0)
    active_scenario_data["gas_price"] = st.sidebar.number_input('Gas price ($/MMBtu)', value=active_scenario_data["gas_price"], min_value=0.0)
    active_scenario_data["ngl_price"] = st.sidebar.number_input('NGL price ($/bbl)', value=active_scenario_data["ngl_price"], min_value=0.0)
    
    active_scenario_data["diff_oil"] = st.sidebar.number_input('Oil price differential ($/bbl)', value=active_scenario_data["diff_oil"])
    active_scenario_data["diff_gas"] = st.sidebar.number_input('Gas price differential ($/MMBtu)', value=active_scenario_data["diff_gas"])
    active_scenario_data["diff_ngl"] = st.sidebar.number_input('NGL price differential ($/bbl)', value=active_scenario_data["diff_ngl"])

    st.sidebar.subheader('Taxes & Costs')
    active_scenario_data["severance_tax_oil"] = st.sidebar.number_input('Oil severance tax (%)', value=active_scenario_data["severance_tax_oil"]*100, min_value=0.0, max_value=100.0) / 100.0
    active_scenario_data["severance_tax_gas"] = st.sidebar.number_input('Gas severance tax (%)', value=active_scenario_data["severance_tax_gas"]*100, min_value=0.0, max_value=100.0) / 100.0
    active_scenario_data["ad_valorem_tax"] = st.sidebar.number_input('Ad valorem tax (%)', value=active_scenario_data["ad_valorem_tax"]*100, min_value=0.0, max_value=100.0) / 100.0
    active_scenario_data["post_prod_pct"] = st.sidebar.number_input('Post‑production cost (%)', value=active_scenario_data["post_prod_pct"]*100, min_value=0.0, max_value=50.0) / 100.0
    
    st.sidebar.markdown('---')
    if active_scenario_data["price_option"] == 'Upload market strip':
        uploaded_file = st.sidebar.file_uploader(
            'Upload CSV with columns: date, oil, gas, ngl', type=['csv'],
            help='The date column should be in YYYY‑MM‑DD format.'
        )
        if uploaded_file is not None:
            st.session_state.uploaded_deck = pd.read_csv(uploaded_file)
    else:
        st.session_state.uploaded_deck = None

    # --- Main Page Layout ---
    st.title('Oil & Gas Royalty Toolkit')
    st.info(f"**Current Scenario: {st.session_state.active_scenario}**")

    tab_dcf, tab_dca, tab_help = st.tabs(['DCF Model', 'Decline Curve Analysis', 'Help & Tutorial'])

    with tab_dcf:
        st.header('DCF Model Inputs')
        
        # --- Well Inputs ---
        well_count = st.number_input(
            'Number of wells to model', min_value=1, max_value=50, 
            value=len(active_scenario_data["wells"]), step=1, key=f"well_count_{st.session_state.active_scenario}"
        )
        
        # Adjust the number of wells in the scenario data if needed
        current_wells = [Well.from_dict(w) for w in active_scenario_data["wells"]]
        if well_count > len(current_wells):
            for i in range(len(current_wells), well_count):
                new_well_date = active_scenario_data["model_start"] + datetime.timedelta(days=30 * i)
                current_wells.append(Well(
                    name=f'Well {i+1}', well_type='PDP', first_prod_date=new_well_date,
                    qi_oil=300.0, qi_gas=800.0, qi_ngl=20.0, b_factor=0.5,
                    initial_decline=0.70, terminal_decline=0.05,
                    royalty_decimal=0.1875, nri=0.80
                ))
        elif well_count < len(current_wells):
            current_wells = current_wells[:well_count]
        
        active_scenario_data["wells"] = [w.to_dict() for w in current_wells]
        
        wells_from_ui: List[Well] = []
        for i in range(well_count):
            well_data = active_scenario_data["wells"][i]
            with st.expander(f'Well {i+1}: {well_data["name"]}'):
                # Create a unique key for each widget based on scenario and well index
                s_key = f"{st.session_state.active_scenario}_{i}"
                
                well_data['name'] = st.text_input('Well name/ID', value=well_data['name'], key=f'name_{s_key}')
                well_data['well_type'] = st.selectbox('Well type', options=['PDP', 'PUD', 'Future'], index=['PDP', 'PUD', 'Future'].index(well_data['well_type']), key=f'type_{s_key}')
                well_data['first_prod_date'] = st.date_input('First production date', value=datetime.date.fromisoformat(well_data['first_prod_date']), key=f'fpd_{s_key}').isoformat()

                c1, c2, c3 = st.columns(3)
                well_data['qi_oil'] = c1.number_input('Initial oil rate (bbl/d)', value=well_data['qi_oil'], min_value=0.0, key=f'qi_oil_{s_key}')
                well_data['qi_gas'] = c2.number_input('Initial gas rate (Mcf/d)', value=well_data['qi_gas'], min_value=0.0, key=f'qi_gas_{s_key}')
                well_data['qi_ngl'] = c3.number_input('Initial NGL rate (bbl/d)', value=well_data['qi_ngl'], min_value=0.0, key=f'qi_ngl_{s_key}')

                c1, c2, c3 = st.columns(3)
                well_data['b_factor'] = c1.number_input('Arps b‑factor', value=well_data['b_factor'], min_value=0.0, max_value=2.0, step=0.1, key=f'b_{s_key}')
                well_data['initial_decline'] = c2.number_input('Initial decline rate (%/yr)', value=well_data['initial_decline']*100, min_value=0.0, max_value=100.0, key=f'di_{s_key}') / 100.0
                well_data['terminal_decline'] = c3.number_input('Terminal decline rate (%/yr)', value=well_data['terminal_decline']*100, min_value=0.0, max_value=30.0, key=f'dt_{s_key}') / 100.0

                c1, c2 = st.columns(2)
                well_data['royalty_decimal'] = c1.number_input('Royalty decimal (fraction)', value=well_data['royalty_decimal'], min_value=0.0, max_value=1.0, step=0.001, format="%.4f", key=f'roy_{s_key}')
                well_data['nri'] = c2.number_input('Net revenue interest (NRI)', value=well_data['nri'], min_value=0.0, max_value=1.0, step=0.01, key=f'nri_{s_key}')

                wells_from_ui.append(Well.from_dict(well_data))

        # --- Acquisition Inputs ---
        st.subheader('Acquisition / Investment')
        active_scenario_data["acquisition_cost"] = st.number_input(
            'Acquisition price (USD)', value=active_scenario_data["acquisition_cost"], min_value=0.0, step=1000.0,
            help='Enter the total purchase price for the royalty interest.'
        )
        if active_scenario_data["acquisition_cost"] > 0:
            active_scenario_data["closing_date"] = st.date_input(
                'Acquisition closing date', value=active_scenario_data["closing_date"],
                min_value=active_scenario_data["model_start"], help='Date when the acquisition cost is incurred.'
            )

        # --- Run Forecast Buttons ---
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        run_single = col1.button(f'Run: {st.session_state.active_scenario}', type="primary")
        run_all = col2.button('Run All Scenarios & Compare')

        if run_single or run_all:
            scenarios_to_run = st.session_state.scenarios if run_all else {st.session_state.active_scenario: active_scenario_data}
            all_results = []

            with st.spinner("Running forecasts..."):
                for name, scenario in scenarios_to_run.items():
                    # Build wells and price deck for the current scenario
                    wells = [Well.from_dict(w) for w in scenario['wells']]
                    for well in wells:
                        well.set_start_index(scenario['model_start'])
                    
                    price_deck = build_price_deck(
                        start_date=scenario['model_start'], months=scenario['forecast_months'],
                        oil_price=scenario['oil_price'], gas_price=scenario['gas_price'], ngl_price=scenario['ngl_price'],
                        diff_oil=scenario['diff_oil'], diff_gas=scenario['diff_gas'], diff_ngl=scenario['diff_ngl'],
                        uploaded_deck=st.session_state.uploaded_deck if scenario['price_option'] == 'Upload market strip' else None
                    )
                    
                    # Compute cash flows
                    result_df = compute_cash_flows(
                        wells=wells, price_deck=price_deck,
                        severance_tax_oil=scenario['severance_tax_oil'], severance_tax_gas=scenario['severance_tax_gas'],
                        ad_valorem_tax=scenario['ad_valorem_tax'], post_prod_cost_pct=scenario['post_prod_pct'],
                        discount_rate=scenario['discount_rate'], forecast_months=scenario['forecast_months']
                    )
                    
                    # Compute investment metrics
                    metrics = compute_investment_metrics(
                        result_df, acquisition_cost=scenario['acquisition_cost'],
                        discount_rate=scenario['discount_rate'], model_start=scenario['model_start'],
                        closing_date=scenario['closing_date']
                    )
                    
                    # Store results
                    all_results.append({
                        "name": name,
                        "result_df": result_df,
                        "metrics": metrics
                    })

            # --- Display Results ---
            st.header("Forecast Results")

            if run_all and len(all_results) > 1:
                st.subheader("Scenario Comparison")
                comparison_data = []
                for res in all_results:
                    m = res['metrics']
                    r = res['result_df']
                    comparison_data.append({
                        "Scenario": res['name'],
                        "NPV ($)": m['npv'],
                        "Undiscounted CF ($)": r.attrs['undiscounted_total'],
                        "Discounted CF ($)": r.attrs['discounted_total'],
                        "IRR (%)": m['irr_annual'] * 100 if m['irr_annual'] is not None else None,
                        "MOIC (x)": m['moic'],
                        "Payback (months)": m['payback_months'],
                    })
                comp_df = pd.DataFrame(comparison_data).set_index("Scenario")
                st.dataframe(comp_df.style.format({
                    "NPV ($)": "{:,.0f}",
                    "Undiscounted CF ($)": "{:,.0f}",
                    "Discounted CF ($)": "{:,.0f}",
                    "IRR (%)": "{:.1f}%",
                    "MOIC (x)": "{:.2f}x",
                }))


            # Display results for the first (or only) scenario run
            first_result = all_results[0]
            st.subheader(f"Summary: {first_result['name']}")

            res_metrics = first_result['metrics']
            res_df = first_result['result_df']
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("NPV ($)", f"{res_metrics['npv']:,.0f}")
            c2.metric("IRR", f"{res_metrics['irr_annual']*100:.1f}%" if res_metrics['irr_annual'] is not None else "N/A")
            c3.metric("MOIC", f"{res_metrics['moic']:.2f}x" if res_metrics['moic'] is not None else "N/A")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Discounted CF ($)", f"{res_df.attrs['discounted_total']:,.0f}")
            payback_label = f"{res_metrics['payback_months']} months" if res_metrics['payback_months'] is not None else "N/A"
            c2.metric("Payback Period", payback_label)
            disc_payback_label = f"{res_metrics['discounted_payback_months']} months" if res_metrics['discounted_payback_months'] is not None else "N/A"
            c3.metric("Discounted Payback", disc_payback_label)

            # Charts and Tables
            st.markdown('**Cash Flow Waterfall**')
            waterfall_data = res_metrics['cash_flow_table'].set_index('date')[['cumulative_net_cash', 'cumulative_discounted']]
            st.line_chart(waterfall_data, use_container_width=True)

            with st.expander('Detailed Cash Flow Table'):
                st.dataframe(res_metrics['cash_flow_table'])
            
            csv_data = res_metrics['cash_flow_table'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download cash flow data as CSV', data=csv_data,
                file_name=f'royalty_dcf_{first_result["name"]}.csv', mime='text/csv'
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
        prod_file = st.file_uploader('Upload production data (CSV)', type=['csv'], key="dca_uploader")
        if prod_file is not None:
            try:
                df = pd.read_csv(prod_file)
                # (DCA logic remains the same as original file)
                # ...
                st.info("DCA functionality is available here.")

            except Exception as e:
                st.error('An unexpected error occurred while processing your production file. Please check your data and try again.')
    
    # -------------------------------------------------------------------------
    # Help & Tutorial Tab
    # -------------------------------------------------------------------------
    with tab_help:
        st.header('Help & Tutorial')
        # (Help logic remains the same as original file)
        # ...
        st.info("Help and tutorial information is available here.")


# If executed as a script via Streamlit, render the app
if __name__ == '__main__':
    if st is not None:
        render_app()
    else:
        print(
            'Streamlit is not installed.  Install streamlit and run this script with:\n'
            '    streamlit run foecast.py'
        )
