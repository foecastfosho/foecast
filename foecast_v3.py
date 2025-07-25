"""
Streamlit application for evaluating oil & gas royalty interests using a
discounted cash‑flow (DCF) framework with integrated decline curve
analysis (DCA).  This version includes a multi‑stream DCA module that
allows users to upload production histories containing separate
commodities (e.g. oil_rate, gas_rate, ngl_rate) and fit decline curves
individually for each stream.  The application also provides a
simplified DCF forecasting tool and a help/tutorial section.

The DCF model forecasts monthly production for each well using
hyperbolic decline with an exponential tail and computes gross and net
royalty revenue based on user‑supplied prices, tax rates and royalty
fractions.  It then discounts the cash flows at user‑defined rates.

The DCA tab fits exponential, hyperbolic or harmonic decline models
using non‑linear least squares.  Users can select which commodity
streams to analyse when multiple rate columns are present in the
uploaded data.  Results include fitted parameters (qi, Di, b) and
plots comparing observed versus fitted rates.

To run the app locally:

    streamlit run dcf_model_web_app.py

This file is designed to be self‑contained; no external modules are
required beyond pandas, numpy, matplotlib and scipy.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def exponential_decline(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Exponential decline model: q = qi * exp(-D_i * t).

    Parameters
    ----------
    t : np.ndarray
        Time in months since first production.
    qi : float
        Initial production rate at t=0.
    Di : float
        Initial decline rate per month.

    Returns
    -------
    np.ndarray
        Forecasted production rates at time t.
    """
    return qi * np.exp(-Di * t)


def hyperbolic_decline(t: np.ndarray, qi: float, Di: float, b: float) -> np.ndarray:
    """Hyperbolic decline model: q = qi / (1 + b * D_i * t)^(1/b).

    When b → 0 this tends toward exponential decline.  For b = 1 the
    relationship corresponds to harmonic decline.
    """
    return qi / np.power(1.0 + b * Di * t, 1.0 / b)


def harmonic_decline(t: np.ndarray, qi: float, Di: float) -> np.ndarray:
    """Harmonic decline model: q = qi / (1 + D_i * t).

    This is a special case of the hyperbolic model with b=1.
    """
    return qi / (1.0 + Di * t)


def fit_decline_model(
    t: np.ndarray,
    q_vals: np.ndarray,
    model: str,
    b_override: float | None = None,
    min_terminal_decline: float = 0.05,
) -> dict:
    """Fit a decline curve to observed data using non‑linear least squares.

    Parameters
    ----------
    t : np.ndarray
        Time array (months since first production).
    q_vals : np.ndarray
        Observed production rates.
    model : str
        Type of decline curve ('Exponential', 'Hyperbolic', or 'Harmonic').
    b_override : float | None
        Optional override for b‑factor when using the hyperbolic model.
    min_terminal_decline : float
        Minimum terminal decline (fraction per month) used to cap
        unrealistic declines during optimisation.

    Returns
    -------
    dict
        Dictionary containing fitted parameters, predicted rates, R² and RMSE.
    """
    # Remove any NaNs or infinite values
    mask = np.isfinite(t) & np.isfinite(q_vals)
    t = t[mask]
    q_vals = q_vals[mask]
    if len(t) < 3:
        return {'error': 'Not enough data points for curve fitting.'}

    # Initial guesses and bounds
    qi_guess = max(q_vals[0], 1e-3)
    # Estimate decline based on first two points; fall back to 0.5 if undefined
    if len(t) > 1 and q_vals[1] > 0 and q_vals[0] > 0:
        Di_guess = max(0.0, -np.log(q_vals[1] / q_vals[0]) / (t[1] - t[0]))
    else:
        Di_guess = 0.5

    bounds_lower = [0.0, 0.0]
    bounds_upper = [np.inf, 10.0]

    if model == 'Exponential':
        # Only qi and Di
        try:
            popt, _ = curve_fit(
                exponential_decline, t, q_vals,
                p0=[qi_guess, max(Di_guess, 1e-3)],
                bounds=([0, 1e-5], [np.inf, 10.0])
            )
            qi, Di = popt
            q_pred = exponential_decline(t, qi, Di)
            # Compute statistics
            residuals = q_vals - q_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((q_vals - np.mean(q_vals))**2)
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            RMSE = np.sqrt(np.mean(residuals**2))
            return {
                'qi': float(qi),
                'Di': float(Di),
                'b': 0.0,
                'q_pred': q_pred,
                'R2': R2,
                'RMSE': RMSE,
            }
        except Exception as e:
            return {'error': str(e)}

    elif model == 'Harmonic':
        # For harmonic decline, b=1 so we fit qi and Di with harmonic function
        try:
            popt, _ = curve_fit(
                harmonic_decline, t, q_vals,
                p0=[qi_guess, max(Di_guess, 1e-3)],
                bounds=([0, 1e-5], [np.inf, 10.0])
            )
            qi, Di = popt
            q_pred = harmonic_decline(t, qi, Di)
            residuals = q_vals - q_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((q_vals - np.mean(q_vals))**2)
            R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            RMSE = np.sqrt(np.mean(residuals**2))
            return {
                'qi': float(qi),
                'Di': float(Di),
                'b': 1.0,
                'q_pred': q_pred,
                'R2': R2,
                'RMSE': RMSE,
            }
        except Exception as e:
            return {'error': str(e)}

    else:  # Hyperbolic
        # If b_override is provided, fix b; otherwise fit b as a free parameter
        if b_override is not None:
            def hyperbolic_fixed_b(t, qi, Di):
                return hyperbolic_decline(t, qi, Di, b_override)
            try:
                popt, _ = curve_fit(
                    hyperbolic_fixed_b, t, q_vals,
                    p0=[qi_guess, max(Di_guess, 1e-3)],
                    bounds=([0, 1e-5], [np.inf, 10.0])
                )
                qi_fit, Di_fit = popt
                q_pred = hyperbolic_decline(t, qi_fit, Di_fit, b_override)
                residuals = q_vals - q_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((q_vals - np.mean(q_vals))**2)
                R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                RMSE = np.sqrt(np.mean(residuals**2))
                return {
                    'qi': float(qi_fit),
                    'Di': float(Di_fit),
                    'b': float(b_override),
                    'q_pred': q_pred,
                    'R2': R2,
                    'RMSE': RMSE,
                }
            except Exception as e:
                return {'error': str(e)}
        else:
            # Fit qi, Di and b simultaneously
            def hyperbolic_free_b(t, qi, Di, b):
                # Bound b between 0 and 2 to avoid extreme values
                return hyperbolic_decline(t, qi, Di, b)
            try:
                popt, _ = curve_fit(
                    hyperbolic_free_b, t, q_vals,
                    p0=[qi_guess, max(Di_guess, 1e-3), 0.5],
                    bounds=([0, 1e-5, 0.0], [np.inf, 10.0, 2.0])
                )
                qi_fit, Di_fit, b_fit = popt
                q_pred = hyperbolic_decline(t, qi_fit, Di_fit, b_fit)
                residuals = q_vals - q_pred
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((q_vals - np.mean(q_vals))**2)
                R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                RMSE = np.sqrt(np.mean(residuals**2))
                return {
                    'qi': float(qi_fit),
                    'Di': float(Di_fit),
                    'b': float(b_fit),
                    'q_pred': q_pred,
                    'R2': R2,
                    'RMSE': RMSE,
                }
            except Exception as e:
                return {'error': str(e)}


def forecast_well(
    start_date: datetime.date,
    months: int,
    qi: float,
    Di: float,
    b: float,
    D_term: float,
    first_production_date: datetime.date,
) -> np.ndarray:
    """Generate a monthly forecast for a single well using hyperbolic decline.

    Parameters
    ----------
    start_date : datetime.date
        Start date of the forecast for the entire model.
    months : int
        Number of months to forecast.
    qi, Di, b : float
        Decline curve parameters (initial rate, decline rate per month, b exponent).
    D_term : float
        Terminal decline rate per month applied once the effective decline
        falls below D_term.
    first_production_date : datetime.date
        The date when this well begins producing.

    Returns
    -------
    np.ndarray
        An array of length `months` containing the forecast production
        volumes.  Months before first production are zeros.
    """
    forecast = np.zeros(months)
    # Compute the offset in months between model start and well start
    offset = (first_production_date.year - start_date.year) * 12 + (first_production_date.month - start_date.month)
    if offset < 0:
        offset = 0
    for m in range(offset, months):
        t = m - offset
        # Hyperbolic decline rate as a function of time
        if b > 0:
            # Effective decline at time t
            D_eff = Di / (1.0 + b * Di * t)
        else:
            D_eff = Di
        # Switch to exponential tail if effective decline falls below terminal
        if D_eff <= D_term:
            # Determine time when tail begins
            if b > 0:
                t_tail = (1.0 / D_term - 1.0 / Di) / b
            else:
                t_tail = (np.log(Di / D_term)) / Di
            # Rate at the start of the exponential tail
            q_tail_start = hyperbolic_decline(t_tail, qi, Di, b) if b > 0 else exponential_decline(t_tail, qi, Di)
            # Exponential decay from that point
            dt = t - t_tail
            rate = q_tail_start * np.exp(-D_term * dt)
        else:
            # Hyperbolic decline
            if model := b > 0:
                rate = hyperbolic_decline(t, qi, Di, b)
            else:
                rate = exponential_decline(t, qi, Di)
        forecast[m] = max(rate, 0.0)
    return forecast


def run_dcf_forecast(well_inputs: list[dict], price_inputs: dict, tax_inputs: dict, global_inputs: dict) -> pd.DataFrame:
    """Compute monthly cash flows for all wells and return a DataFrame.

    Parameters
    ----------
    well_inputs : list of dict
        Each dict should contain fields: name, type, first_prod, qi_oil, qi_gas,
        qi_ngl, b, Di, D_term, royalty_decimal, nri.
    price_inputs : dict
        Contains flat prices for oil, gas and NGL (keys: 'oil', 'gas', 'ngl').
    tax_inputs : dict
        Contains severance and ad valorem tax rates (keys: 'severance_oil',
        'severance_gas', 'severance_ngl', 'ad_valorem').  Values should be
        fractions (e.g. 0.046 for 4.6%).
    global_inputs : dict
        Contains model start_date (datetime.date), forecast_horizon (int
        months), discount_rates (list of floats, e.g. [0.1, 0.15]).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by month number with columns for gross and net
        revenue, cash flow, discounted cash flows and cumulative cash flow.
    """
    start_date = global_inputs['start_date']
    months = global_inputs['forecast_horizon']
    discount_rates = global_inputs['discount_rates']

    # Initialize arrays to accumulate production across wells
    oil_total = np.zeros(months)
    gas_total = np.zeros(months)
    ngl_total = np.zeros(months)
    # Track net royalty fraction per well (royalty_decimal * nri)
    royalty_shares = []
    for w in well_inputs:
        # Forecast production for each commodity
        oil_forecast = forecast_well(start_date, months, w['qi_oil'], w['Di'], w['b'], w['D_term'], w['first_prod'])
        gas_forecast = forecast_well(start_date, months, w['qi_gas'], w['Di'], w['b'], w['D_term'], w['first_prod'])
        ngl_forecast = forecast_well(start_date, months, w['qi_ngl'], w['Di'], w['b'], w['D_term'], w['first_prod'])
        oil_total += oil_forecast
        gas_total += gas_forecast
        ngl_total += ngl_forecast
        royalty_shares.append(w['royalty_decimal'] * w['nri'])

    # Compute average royalty share across wells weighted by production
    royalty_share = np.mean(royalty_shares) if royalty_shares else 0.0

    # Compute revenue: price * volume
    price_oil = price_inputs['oil']
    price_gas = price_inputs['gas']
    price_ngl = price_inputs['ngl']
    gross_rev_oil = oil_total * price_oil
    gross_rev_gas = gas_total * price_gas
    gross_rev_ngl = ngl_total * price_ngl
    gross_rev = gross_rev_oil + gross_rev_gas + gross_rev_ngl

    # Apply severance taxes
    net_after_sev = gross_rev.copy()
    if tax_inputs:
        net_after_sev -= (
            gross_rev_oil * tax_inputs.get('severance_oil', 0.0)
            + gross_rev_gas * tax_inputs.get('severance_gas', 0.0)
            + gross_rev_ngl * tax_inputs.get('severance_ngl', 0.0)
        )
    # Apply ad valorem tax on net revenue
    net_after_taxes = net_after_sev * (1.0 - tax_inputs.get('ad_valorem', 0.0))

    # Royalty owner's net revenue
    net_royalty = net_after_taxes * royalty_share

    # Discount cash flows
    df = pd.DataFrame({'Month': np.arange(1, months + 1)})
    df['NetRoyalty'] = net_royalty
    # Cumulative cash flow for payback
    df['Cumulative'] = df['NetRoyalty'].cumsum()
    # Discounted cash flows for each rate
    for rate in discount_rates:
        discount_factor = 1.0 / np.power(1.0 + rate / 12.0, df['Month'])
        df[f'PV_{int(rate * 100)}'] = df['NetRoyalty'] * discount_factor
    return df


def compute_summary_metrics(df: pd.DataFrame, discount_rates: list[float]) -> dict:
    """Compute undiscounted cash flow, PV values and payback period from cash flow DataFrame."""
    metrics = {}
    total_cf = df['NetRoyalty'].sum()
    metrics['Total (Undiscounted)'] = total_cf
    for rate in discount_rates:
        pv_col = f'PV_{int(rate * 100)}'
        metrics[f'PV{int(rate * 100)}'] = df[pv_col].sum()
    # Payback period: first month where cumulative >= 0
    payback_month = df.index[df['Cumulative'] >= 0.0].min()
    if pd.isna(payback_month):
        metrics['Payback (months)'] = 'Never'
    else:
        metrics['Payback (months)'] = int(payback_month) + 1
    return metrics


def main() -> None:
    st.set_page_config(page_title='Oil & Gas Royalty DCF Model', layout='wide')
    st.title('Oil & Gas Royalty DCF Model')

    tab_dcf, tab_dca, tab_help = st.tabs(['DCF Model', 'Decline Curve Analysis', 'Help & Tutorial'])

    # ---------------------------------------------------------------------
    # DCF MODEL TAB
    # ---------------------------------------------------------------------
    with tab_dcf:
        st.header('DCF Model')
        st.markdown('Define your wells and forecast economics based on decline curves, prices and taxes.')

        # Global assumptions
        with st.expander('Global Parameters', expanded=True):
            start_date = st.date_input('Model start date', datetime.date.today().replace(day=1))
            horizon_years = st.number_input('Forecast horizon (years)', value=15, min_value=1, max_value=40)
            forecast_horizon = horizon_years * 12
            # Prices
            st.subheader('Commodity Prices (Flat)')
            price_oil = st.number_input('Oil price ($/bbl)', value=70.0, min_value=0.0)
            price_gas = st.number_input('Gas price ($/Mcf)', value=3.5, min_value=0.0)
            price_ngl = st.number_input('NGL price ($/bbl)', value=25.0, min_value=0.0)
            # Taxes
            st.subheader('Taxes')
            severance_oil = st.number_input('Severance tax – Oil (%)', value=4.6, min_value=0.0, max_value=20.0) / 100.0
            severance_gas = st.number_input('Severance tax – Gas (%)', value=7.5, min_value=0.0, max_value=20.0) / 100.0
            severance_ngl = st.number_input('Severance tax – NGL (%)', value=4.6, min_value=0.0, max_value=20.0) / 100.0
            ad_valorem = st.number_input('Ad valorem tax (%)', value=1.0, min_value=0.0, max_value=10.0) / 100.0
            # Discount rates
            st.subheader('Discount Rates')
            discount_rates = []
            for lbl, default in [('PV10', 10.0), ('PV15', 15.0)]:
                rate = st.number_input(f'{lbl} (% per year)', value=default, min_value=0.0, max_value=50.0)
                discount_rates.append(rate / 100.0)

        # Number of wells
        num_wells = st.number_input('Number of wells', value=1, min_value=1, max_value=20, step=1)
        well_inputs: list[dict] = []
        if num_wells > 0:
            for i in range(int(num_wells)):
                with st.expander(f'Well {i+1}', expanded=(i == 0)):
                    name = st.text_input(f'Well {i+1} name', value=f'Well_{i+1}')
                    well_type = st.selectbox(f'Well {i+1} type', options=['PDP', 'PUD', 'Future'], index=0)
                    first_prod = st.date_input(f'First production date (Well {i+1})', start_date)
                    qi_oil = st.number_input(f'Initial oil rate (bbl/day) – Well {i+1}', value=200.0, min_value=0.0)
                    qi_gas = st.number_input(f'Initial gas rate (Mcf/day) – Well {i+1}', value=500.0, min_value=0.0)
                    qi_ngl = st.number_input(f'Initial NGL rate (bbl/day) – Well {i+1}', value=50.0, min_value=0.0)
                    b_factor = st.number_input(f'b‑factor – Well {i+1}', value=0.5, min_value=0.0, max_value=2.0, step=0.1)
                    Di = st.number_input(f'Initial decline rate (per year) – Well {i+1}', value=0.70, min_value=0.0, max_value=1.0, step=0.01)
                    Di_month = Di / 12.0
                    D_term = st.number_input(f'Terminal decline rate (per year) – Well {i+1}', value=0.10, min_value=0.0, max_value=1.0, step=0.01) / 12.0
                    royalty_decimal = st.number_input(f'Royalty decimal – Well {i+1}', value=0.125, min_value=0.0, max_value=1.0, step=0.001)
                    nri = st.number_input(f'Net revenue interest (NRI) – Well {i+1}', value=1.0, min_value=0.0, max_value=1.0, step=0.001)
                    well_inputs.append({
                        'name': name,
                        'type': well_type,
                        'first_prod': first_prod,
                        'qi_oil': qi_oil,
                        'qi_gas': qi_gas,
                        'qi_ngl': qi_ngl,
                        'b': b_factor,
                        'Di': Di_month,
                        'D_term': D_term,
                        'royalty_decimal': royalty_decimal,
                        'nri': nri,
                    })

        if st.button('Run forecast'):
            price_inputs = {'oil': price_oil, 'gas': price_gas, 'ngl': price_ngl}
            tax_inputs = {
                'severance_oil': severance_oil,
                'severance_gas': severance_gas,
                'severance_ngl': severance_ngl,
                'ad_valorem': ad_valorem,
            }
            global_inputs = {
                'start_date': start_date,
                'forecast_horizon': forecast_horizon,
                'discount_rates': discount_rates,
            }
            df_cf = run_dcf_forecast(well_inputs, price_inputs, tax_inputs, global_inputs)
            summary = compute_summary_metrics(df_cf, discount_rates)
            st.subheader('Summary Metrics')
            for key, val in summary.items():
                st.write(f'{key}: {val:,.2f}' if isinstance(val, (int, float)) else f'{key}: {val}')
            st.subheader('Cash Flow Table')
            st.dataframe(df_cf)
            # Plot cumulative cash flow
            fig_cf, ax_cf = plt.subplots()
            ax_cf.plot(df_cf['Month'], df_cf['Cumulative'], label='Cumulative Net Royalty')
            ax_cf.set_xlabel('Month')
            ax_cf.set_ylabel('Cumulative Net Royalty ($)')
            ax_cf.set_title('Cumulative Net Royalty by Month')
            ax_cf.grid(True)
            st.pyplot(fig_cf)

    # ---------------------------------------------------------------------
    # DECLINE CURVE ANALYSIS TAB
    # ---------------------------------------------------------------------
    with tab_dca:
        st.header('Decline Curve Analysis (DCA)')
        st.markdown(
            'Upload historical production data to estimate decline curve parameters.\n'
            'The CSV should contain a `date` column (YYYY‑MM‑DD), a `well_id` column, and one or more rate columns.\n'
            'If you include columns ending with `_rate` (e.g. `oil_rate`, `gas_rate`), the app will treat each of those as a separate stream and allow you to fit decline curves individually.'
        )
        uploaded_file = st.file_uploader('Upload production history CSV', type=['csv'])
        if uploaded_file is not None:
            try:
                df_prod = pd.read_csv(uploaded_file)
                # Ensure required columns exist
                if 'date' not in df_prod.columns or 'well_id' not in df_prod.columns:
                    st.error('CSV must contain `date` and `well_id` columns.')
                else:
                    # Convert dates and sort
                    df_prod['date'] = pd.to_datetime(df_prod['date'])
                    df_prod.sort_values(['well_id', 'date'], inplace=True)
                    # Compute time since first production per well in months
                    df_prod['t_months'] = (
                        df_prod.groupby('well_id')['date']
                        .transform(lambda s: (s - s.min()).dt.days / 30.4375)
                    )
                    # Identify rate columns
                    rate_columns = [col for col in df_prod.columns if col.endswith('_rate')]
                    if not rate_columns and 'rate' in df_prod.columns:
                        rate_columns = ['rate']
                    if not rate_columns:
                        st.error('No rate columns found. Include a `rate` column or columns ending with `_rate`.')
                    else:
                        # Select well and streams
                        well_ids = df_prod['well_id'].unique().tolist()
                        selected_well = st.selectbox('Select well', well_ids)
                        selected_streams = st.multiselect(
                            'Select stream(s) to analyse', options=rate_columns, default=rate_columns
                        )
                        model_choice = st.selectbox('Decline model', ['Exponential', 'Hyperbolic', 'Harmonic'], index=1)
                        b_override = None
                        if model_choice == 'Hyperbolic':
                            if st.checkbox('Manual b‑factor override', value=False):
                                b_override = st.number_input('b‑factor (override)', value=0.5, min_value=0.0, max_value=2.0, step=0.1)
                        min_decl_per_year = st.number_input('Minimum terminal decline (% per year)', value=5.0, min_value=0.0, max_value=30.0) / 100.0
                        min_decl = min_decl_per_year / 12.0
                        # Loop over each selected stream
                        for stream in selected_streams:
                            st.markdown(f'#### {stream.replace("_rate", "").title()}')
                            well_data = df_prod[df_prod['well_id'] == selected_well].copy()
                            if stream not in well_data.columns:
                                st.warning(f'Stream {stream} not found for the selected well.')
                                continue
                            q_series = well_data[stream].astype(float)
                            t_series = well_data['t_months'].astype(float)
                            # Outlier removal (optional)
                            if st.checkbox(f'Remove outliers for {stream}', value=True):
                                mean_val = q_series.mean()
                                std_val = q_series.std() if q_series.std() > 0 else 1.0
                                mask = np.abs(q_series - mean_val) <= 3.0 * std_val
                                q_vals = q_series[mask].values
                                t_vals = t_series[mask].values
                            else:
                                q_vals = q_series.values
                                t_vals = t_series.values
                            # Remove zero or negative rates
                            positive_mask = q_vals > 0
                            q_vals = q_vals[positive_mask]
                            t_vals = t_vals[positive_mask]
                            result = fit_decline_model(t_vals, q_vals, model_choice, b_override=b_override, min_terminal_decline=min_decl)
                            if 'error' in result:
                                st.error(f"Error fitting {stream}: {result['error']}")
                            else:
                                st.write(f"Initial rate (qᵢ): {result['qi']:.2f}")
                                st.write(f"Initial decline (Dᵢ) per month: {result['Di']:.4f}")
                                st.write(f"b‑factor: {result['b']:.3f}")
                                st.write(f"R²: {result['R2']:.3f}")
                                st.write(f"RMSE: {result['RMSE']:.3f}")
                                # Plot observed vs fitted
                                fig, ax = plt.subplots()
                                ax.plot(t_vals, q_vals, 'o', label='Observed')
                                ax.plot(t_vals, result['q_pred'], '-', label='Fitted')
                                ax.set_xlabel('Time (months)')
                                # Use stream name for y label; remove _rate suffix
                                ylabel = stream.replace('_rate', '').upper()
                                ax.set_ylabel(f'Production Rate ({ylabel})')
                                ax.set_title(f'Decline Curve Fit – {stream.replace("_rate", "").title()} ({model_choice})')
                                ax.legend()
                                st.pyplot(fig)
            except Exception as e:
                st.error(f'Error reading CSV: {e}')
        else:
            st.info('Please upload a CSV file to perform decline curve analysis.')

    # ---------------------------------------------------------------------
    # HELP & TUTORIAL TAB
    # ---------------------------------------------------------------------
    with tab_help:
        st.header('Help & Tutorial')
        st.markdown('### Overview')
        st.markdown(
            'This model evaluates the value of an oil and gas royalty interest using discounted cash flow analysis.\n'
            'It forecasts monthly production from user‑defined wells, applies price assumptions, taxes and costs, and discounts the resulting cash flows to compute metrics such as net present value (NPV) and payback period.\n'
            'The Decline Curve Analysis tab lets you estimate decline parameters (qᵢ, Dᵢ and b) from historical production data.\n'
            'For example, you can assess a 1 % royalty in a 640‑acre unit with multiple producing wells and undrilled locations.'
        )
        # Workflow diagram (optional)
        st.markdown('### Model Workflow')
        flow_path = Path(__file__).with_name('dcf_flow_diagram.png')
        if flow_path.exists():
            try:
                st.image(str(flow_path), caption='High‑level workflow of the DCF model', use_column_width=True)
            except Exception:
                st.info('Workflow diagram image not available.')
        else:
            st.info('Workflow diagram image not found.')
        # Glossary
        st.markdown('### Glossary of Key Terms')
        glossary_items = [
            ('DCF (Discounted Cash Flow)', 'A valuation method estimating the present value of an investment based on its expected future cash flows.'),
            ('NPV / PV10 / PV15', 'Net present value and present values using 10 % and 15 % discount rates, widely used in reserve valuation.'),
            ('qᵢ (Initial Production Rate)', 'The production rate at the start of the forecast, measured in barrels or Mcf per day.'),
            ('Dᵢ (Initial Decline Rate)', 'The decline rate during the early high‑decline phase of production; shale wells often decline 64–70 % in year one.'),
            ('b‑factor (Decline Exponent)', 'Controls the shape of a decline curve: b=0 for exponential, b=1 for harmonic, and 0<b<1 for hyperbolic.'),
            ('NRI (Net Revenue Interest)', 'The portion of production revenue a royalty owner receives after accounting for burdens and overrides.'),
            ('Royalty Decimal', 'The fractional share of production revenue stipulated in a lease, expressed as a decimal (e.g., 0.125).'),
            ('PDP / PUD', 'Proved Developed Producing and Proved Undeveloped reserve categories used to classify well status.'),
            ('Strip Pricing', 'Using market futures prices to create a forward price curve for forecasting.'),
            ('Price Differential', 'The difference between a benchmark price (e.g., WTI or Henry Hub) and the realized price at the lease, reflecting quality and transportation adjustments.'),
            ('Post‑Production Costs', 'Costs incurred after extraction, such as gathering, compression and transportation, that may be deducted from revenue before paying royalties.'),
        ]
        for term, definition in glossary_items:
            st.markdown(f'**{term}**: {definition}')
        st.markdown('### Step‑by‑Step Guide')
        st.markdown(
            '* **Enter Your Wells** – In the DCF Model tab, specify well parameters such as initial rates, decline parameters, royalty fraction and NRI.\n'
            '* **Set Global Assumptions** – Define the start date, forecast horizon, flat commodity prices, tax rates and discount rates.\n'
            '* **Run the Forecast** – Click **Run forecast** to compute monthly production, revenues, taxes and net royalty cash flows, and view summary metrics and plots.\n'
            '* **Perform Decline Analysis** – In the DCA tab, upload historical production data with `date`, `well_id` and rate columns (e.g. `oil_rate`, `gas_rate`), select the well and streams, and fit decline curves.\n'
            '* **Review Results** – Examine fitted parameters (qᵢ, Dᵢ, b), R² and RMSE values, and use them to inform your well inputs in the DCF model.\n'
            '* **Download the Guide** – A printable PDF guide is available below.'
        )
        # Decline curve examples image
        decline_example_path = Path(__file__).with_name('decline_curve_examples.png')
        st.markdown('### Decline Curve Examples')
        if decline_example_path.exists():
            try:
                st.image(str(decline_example_path), caption='Example decline curves: exponential, hyperbolic and harmonic', use_column_width=True)
            except Exception:
                st.info('Example decline curves image not available.')
        else:
            st.info('Decline curve examples image not found.')
        # Downloadable PDF guide
        st.markdown('### Downloadable Guide')
        pdf_path = Path(__file__).with_name('dcf_help_guide.pdf')
        if pdf_path.exists():
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                st.download_button('Download Help Guide (PDF)', data=pdf_bytes, file_name='DCF_Help_Guide.pdf', mime='application/pdf')
            except Exception:
                st.info('Unable to load the PDF guide.')
        else:
            st.info('PDF guide not found in the app directory.')


if __name__ == '__main__':
    main()