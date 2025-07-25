import datetime
from dataclasses import dataclass, field
from typing import List, Optional
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
    # Build price deck
    price_deck = build_price_deck(
        start_date=model_start,
        months=forecast_months,
        oil_price=oil_price,
        gas_price=gas_price,
        ngl_price=ngl_price,
        diff_oil=diff_oil,
        diff_gas=diff_gas,
        diff_ngl=diff_ngl,
        uploaded_deck=uploaded_deck,
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
        # Set start month indices for all wells
        for well in wells:
            well.set_start_index(model_start)
        # Compute cash flows when user clicks button
    if st.button('Run forecast'):
        result_df = compute_cash_flows(
            wells=wells,
            price_deck=price_deck,
            severance_tax_oil=severance_tax_oil,
            severance_tax_gas=severance_tax_gas,
            ad_valorem_tax=ad_valorem_tax,
            post_prod_cost_pct=post_prod_pct,
            discount_rate=discount_rate,
            forecast_months=forecast_months,
        )
        # Display summary metrics
        st.subheader('Summary Results')
        col1, col2, col3 = st.columns(3)
        col1.metric('Undiscounted Net Cash Flow ($)', f"{result_df.attrs['undiscounted_total']:,.0f}")
        col2.metric('Discounted Net Cash Flow ($)', f"{result_df.attrs['discounted_total']:,.0f}")
        payback = result_df.attrs['payback_date']
        col3.metric('Payback Date', payback.strftime('%Y-%m') if payback is not None else 'N/A')
        # PV by well type
        st.markdown('**Present Value by Well Type**')
        pv_type_df = pd.DataFrame(
            {
                'Well type': list(result_df.attrs['pv_by_type'].keys()),
                'PV (USD)': list(result_df.attrs['pv_by_type'].values()),
                'Undiscounted (USD)': list(result_df.attrs['undiscounted_by_type'].values()),
            }
        )
        st.dataframe(pv_type_df)
        # Plot cumulative cash flow
        st.line_chart(
            result_df.set_index('date')[['cumulative_cf', 'cumulative_discounted']],
            use_container_width=True
        )
        # Display monthly cash flow table (optionally limit rows)
        with st.expander('Detailed Cash Flow Table'):
            st.dataframe(result_df)
        # Provide option to download results as CSV
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
                    mc_results = run_monte_carlo(
                        wells=wells,
                        price_deck=price_deck,
                        severance_tax_oil=severance_tax_oil,
                        severance_tax_gas=severance_tax_gas,
                        ad_valorem_tax=ad_valorem_tax,
                        post_prod_cost_pct=post_prod_pct,
                        discount_rate=discount_rate,
                        forecast_months=forecast_months,
                        model_start=model_start,
                        iterations=int(iterations),
                        price_sigma=float(price_sigma),
                        vol_sigma=float(vol_sigma),
                    )
                # Compute percentiles
                p10 = np.percentile(mc_results, 10)
                p50 = np.percentile(mc_results, 50)
                p90 = np.percentile(mc_results, 90)
                st.write(
                    f'**P10 NPV:** {p10:,.0f} USD\n\n'
                    f'**P50 (median) NPV:** {p50:,.0f} USD\n\n'
                    f'**P90 NPV:** {p90:,.0f} USD\n\n'
                    'These values represent the distribution of discounted cash flows across the simulated scenarios.'
                )
                # Plot histogram of results
                hist_df = pd.DataFrame({'NPV': mc_results})
                st.bar_chart(hist_df['NPV'].value_counts().sort_index())
        # Instructions on updating price decks
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
            try:
                # Read the uploaded CSV into a DataFrame
                df = pd.read_csv(prod_file)

                # Help tooltip for accepted date formats
                st.markdown(
                    '<small><em>Accepted date formats include MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, MMM-YY and Excel serial numbers.</em></small>',
                    unsafe_allow_html=True
                )

                # Let the user select the date column
                date_cols = [c for c in df.columns if df[c].dtype == object or 'date' in c.lower()]
                if not date_cols:
                    st.error('No suitable date column found. Please ensure your CSV contains a date column.')
                    st.stop()
                date_col = st.selectbox('Select the date column', date_cols)

                # Option to automatically convert date formats
                auto_convert = st.checkbox('Auto-convert date formats', value=True, help='Attempt to recognize and standardize common date formats.')

                def parse_dates(series, auto, user_fmt=None):
                    """Parse a pandas Series of dates. Attempts multiple formats or a user-specified format."""
                    ser = series.copy()
                    if user_fmt:
                        try:
                            return pd.to_datetime(ser, format=user_fmt, errors='coerce')
                        except Exception:
                            return pd.Series([pd.NaT] * len(ser))
                    if not auto:
                        return pd.Series([pd.NaT] * len(ser))
                    parsed = pd.to_datetime(ser, errors='coerce', infer_datetime_format=True)
                    if parsed.isna().any():
                        parsed_alt = pd.to_datetime(ser, errors='coerce', dayfirst=True, infer_datetime_format=True)
                        if parsed_alt.notna().sum() > parsed.notna().sum():
                            parsed = parsed_alt
                    ser_numeric = pd.to_numeric(ser, errors='coerce')
                    mask_serial = parsed.isna() & ser_numeric.notna()
                    if mask_serial.any():
                        origin = datetime.datetime(1899, 12, 30)
                        parsed.loc[mask_serial] = ser_numeric.loc[mask_serial].apply(lambda x: origin + datetime.timedelta(days=float(x)))
                    return parsed

                parsed_dates = parse_dates(df[date_col], auto_convert)
                unresolved_count = parsed_dates.isna().sum()
                manual_fmt = None
                if unresolved_count > 0:
                    st.warning(f'{unresolved_count} date values could not be automatically parsed.')
                    fmt_options = [
                        ('MM/DD/YYYY', '%m/%d/%Y'),
                        ('DD/MM/YYYY', '%d/%m/%Y'),
                        ('YYYY-MM-DD', '%Y-%m-%d'),
                        ('MMM-YY', '%b-%y'),
                        ('MMM-YYYY', '%b-%Y'),
                    ]
                    fmt_label = st.selectbox('Select a date format to apply manually (optional)', [f[0] for f in fmt_options] + ['None'], index=len(fmt_options))
                    if fmt_label != 'None':
                        manual_fmt = dict(fmt_options).get(fmt_label)
                        parsed_dates = parse_dates(df[date_col], False, manual_fmt)
                preview_df = pd.DataFrame({ 'Original': df[date_col].head(10), 'Parsed': parsed_dates.head(10) })
                st.dataframe(preview_df, height=200)
                if parsed_dates.isna().any():
                    st.error('Some dates could not be parsed. Please adjust the date format or correct the data and try again.')
                    st.stop()

                df = df.copy()
                df['date_parsed'] = parsed_dates

                daily_data = df['date_parsed'].dt.day.nunique() > 1
                aggregate_month = False
                if daily_data:
                    aggregate_month = st.checkbox('Aggregate daily data to monthly totals', value=False)
                if aggregate_month:
                    df['period'] = df['date_parsed'].dt.to_period('M').dt.to_timestamp()
                else:
                    df['period'] = df['date_parsed']

                # Choose the well identifier column
                candidate_cols = [c for c in df.columns if (df[c].dtype == object) or c.lower().endswith('id')]
                if not candidate_cols:
                    st.error('No suitable well identifier column found. Please include a column such as well_id or API.')
                    st.stop()
                default_index = candidate_cols.index('well_id') if 'well_id' in candidate_cols else 0
                well_col = st.selectbox('Select the well identifier column', options=candidate_cols, index=default_index)

                # Identify production rate columns using flexible header recognition
                # Define synonyms for each commodity to match common naming conventions
                synonyms = {
                    'oil': [
                        'oil', 'oil prod', 'oil production', 'oil_rate', 'bo', 'bo/d', 'bopd',
                        'daily oil', 'crude', 'crude oil', 'oilvolume'
                    ],
                    'gas': [
                        'gas', 'gas prod', 'gas production', 'mcf', 'mcfd', 'gasrate', 'gas rate',
                        'methane', 'dry gas'
                    ],
                    'ngl': [
                        'ngl', 'ngl prod', 'ngl production', 'natural gas liquids', 'condensate',
                        'c3+', 'y-grade', 'y grade', 'cond'
                    ],
                }
                # Preprocess column names: remove spaces and underscores, convert to lowercase
                col_processed = {c: c.strip().lower().replace(' ', '').replace('_', '') for c in df.columns}
                # Consider all columns except date-related and identifier columns as candidates for production rates
                candidate_cols = [c for c in df.columns if c not in [date_col, 'date_parsed', 'period', well_col]]
                # Initialize matches dictionary for each commodity
                matches = {key: [] for key in synonyms}
                for col in candidate_cols:
                    col_norm = col_processed[col]
                    for comm, keys in synonyms.items():
                        for key in keys:
                            key_norm = key.replace(' ', '').replace('_', '').lower()
                            if key_norm and key_norm in col_norm:
                                matches[comm].append(col)
                                break  # stop checking further synonyms for this commodity
                # Offer manual override for mapping if desired
                advanced_mapping = st.checkbox(
                    'Advanced Mapping (manually assign production columns)',
                    value=False,
                    help='Enable this to manually assign uploaded columns to oil, gas, and NGL rates.'
                )
                selected_map = {}
                if advanced_mapping:
                    # For each commodity, allow user to pick a column or None
                    for comm in ['oil', 'gas', 'ngl']:
                        options = ['None'] + candidate_cols
                        # Preselect first matched candidate if available
                        pre_index = 0
                        if matches[comm]:
                            first_match = matches[comm][0]
                            if first_match in options:
                                pre_index = options.index(first_match)
                        sel = st.selectbox(
                            f'Select column for {comm.title()} production',
                            options,
                            index=pre_index
                        )
                        if sel != 'None':
                            selected_map[comm] = sel
                else:
                    # Automatic mapping: assign columns with a unique candidate; prompt when multiple candidates exist
                    for comm in ['oil', 'gas', 'ngl']:
                        cands = list(dict.fromkeys(matches[comm]))
                        if len(cands) == 1:
                            selected_map[comm] = cands[0]
                        elif len(cands) > 1:
                            sel = st.selectbox(
                                f'Multiple columns detected for {comm.title()} production; please select one',
                                ['None'] + cands,
                                index=1
                            )
                            if sel != 'None':
                                selected_map[comm] = sel
                # Consolidate selected columns
                rate_columns = list(dict.fromkeys(selected_map.values()))
                # Fallback: if no columns selected via synonyms, use conventional naming patterns
                if not rate_columns:
                    fallback = [c for c in df.columns if c.endswith('_rate')]
                    if not fallback and 'rate' in df.columns:
                        fallback = ['rate']
                    rate_columns = fallback
                # Display mapping summary to the user
                for comm, col in selected_map.items():
                    st.info(f"Mapped '{col}' to {comm.title()} rate")
                # If no rate columns identified at all, inform the user and stop
                if not rate_columns:
                    st.error('No production rate columns were identified. Please rename columns or use Advanced Mapping to assign them.')
                    st.stop()

                # Aggregate by well and period if needed
                if aggregate_month:
                    agg_dict = {col: 'sum' for col in rate_columns}
                    df_grouped = df.groupby([well_col, 'period'], as_index=False).agg(agg_dict)
                    df_grouped['date_parsed'] = df_grouped['period']
                else:
                    df_grouped = df[[well_col, 'date_parsed'] + rate_columns].copy()

                df_grouped.sort_values([well_col, 'date_parsed'], inplace=True)
                df_grouped['t_months'] = df_grouped.groupby(well_col)['date_parsed'].transform(lambda s: (s - s.min()).dt.days / 30.4375)

                # Prompt for well and streams to analyse
                unique_wells = df_grouped[well_col].unique()
                selected_well = st.selectbox('Select well for analysis', unique_wells)
                selected_streams = st.multiselect('Select the streams you want to analyse', options=rate_columns, default=rate_columns)

                well_data = df_grouped[df_grouped[well_col] == selected_well].copy()
                if well_data.empty:
                    st.warning('No data found for the selected well.')
                    st.stop()

                for stream in selected_streams:
                    q_series = well_data[stream].astype(float)
                    remove_outliers = st.checkbox(f'Remove outliers for {stream}', value=True)
                    if remove_outliers:
                        mean = q_series.mean()
                        std = q_series.std() if q_series.std() > 0 else 1.0
                        mask = (abs(q_series - mean) <= 3 * std)
                        t_vals = well_data['t_months'].values[mask]
                        q_vals = q_series.values[mask]
                    else:
                        t_vals = well_data['t_months'].values
                        q_vals = q_series.values
                    mask_pos = q_vals > 0
                    t_vals = t_vals[mask_pos]
                    q_vals = q_vals[mask_pos]
                    model_choice = st.selectbox(f'Decline model for {stream}', ['Exponential', 'Hyperbolic', 'Harmonic'])
                    b_override = None
                    if model_choice == 'Hyperbolic' and st.checkbox(f'Manual b‑factor override for {stream}', value=False):
                        b_override = st.number_input(f'b‑factor (override) for {stream}', value=0.5, min_value=0.0, max_value=2.0, step=0.1)
                    min_dt = st.number_input(f'Minimum terminal decline (%) for {stream}', value=5.0, min_value=0.0, max_value=30.0, step=0.5) / 100.0
                    if len(q_vals) >= 3:
                        result = fit_decline_model(t_vals, q_vals, model_choice, b_override=b_override, min_terminal_decline=min_dt)
                        if 'error' in result:
                            st.error(f"Error fitting curve for {stream}: {result['error']}")
                        else:
                            base_name = stream.replace('_rate', '')
                            st.markdown(f'#### {base_name.title()} decline fit')
                            st.write(f"Initial rate (qᵢ): {result['qi']:.2f}")
                            st.write(f"Initial decline (Dᵢ): {result['Di']:.4f} per month")
                            st.write(f"b‑factor: {result['b']:.3f}")
                            st.write(f"R²: {result['R2']:.3f}")
                            st.write(f"RMSE: {result['RMSE']:.3f}")
                            fig, ax = plt.subplots()
                            ax.plot(t_vals, q_vals, 'o', label=f'Observed {base_name}')
                            ax.plot(t_vals, result['q_pred'], '-', label=f'Fitted {base_name}')
                            ax.set_xlabel('Time (months)')
                            ax.set_ylabel(f'Production rate ({base_name.upper()})')
                            ax.set_title(f'Decline Curve Fit – {base_name.title()} ({model_choice})')
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        st.warning(f'Not enough data points to fit a decline curve for {stream}')
            except Exception as e:
                st.error('An unexpected error occurred while processing your production file. Please check your data and try again.')
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
