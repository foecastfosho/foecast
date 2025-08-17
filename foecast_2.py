"""
Streamlit application for oil & gas royalty forecasting.

This simplified version of ``foecast_2.py`` incorporates numerous
improvements over the original implementation:

* Separation of the computational engine into ``forecast_model.py`` for
  improved modularity and testability.
* Deep copying of scenarios to prevent shared state between scenarios.
* Unified naming of the post‑production cost fraction (``post_prod_cost_pct``)
  and backward‑compatible reading of the older ``post_prod_pct`` key.
* Protection against non‑physical b‑factor values by treating b≈0 as
  exponential decline (implemented in ``forecast_model``).
* Enforcement of logical constraints between initial and terminal decline
  rates in the UI.
* Clamping of negative realized prices to zero when differentials or
  deductions exceed the index price (handled in ``forecast_model``).

The focus of this rewrite is on the DCF model; the original Decline
Curve Analysis (DCA) tab and extensive UI styling have been omitted for
brevity. This script should serve as a clear starting point for
customising and extending the royalty toolkit while adhering to best
practices in code organisation and robustness.
"""

from __future__ import annotations

import datetime
import copy
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

try:
    import streamlit as st  # type: ignore
except ImportError:
    st = None

# ---------------------------------------------------------------------------
# Helper functions

def render_dca_and_help() -> None:
    """
    Render the Decline Curve Analysis (DCA) and Help & Tutorial sections.

    This helper encapsulates the logic for uploading production data, fitting
    decline curves, plotting the results, and displaying usage guidance.
    Calling this function ensures the DCA tools and help text are available
    wherever it is invoked.
    """
    # Decline Curve Analysis section
    st.header("Decline Curve Analysis")
    dca_file = st.file_uploader(
        "Upload CSV for decline curve analysis",
        type=["csv"],
        help="The CSV must include a 'date' column and at least one numeric rate column."
    )
    if dca_file is not None:
        try:
            dca_df = pd.read_csv(dca_file)
            # Ensure date column exists
            if "date" not in dca_df.columns:
                st.error("The uploaded file must include a 'date' column.")
            else:
                dca_df["date"] = pd.to_datetime(dca_df["date"])
                # Select numeric rate columns (exclude date)
                numeric_cols = [col for col in dca_df.select_dtypes(include=[np.number]).columns if col != "date"]
                if len(numeric_cols) == 0:
                    st.error("The uploaded file must include at least one numeric rate column.")
                else:
                    rate_col = st.selectbox("Select rate column", options=numeric_cols)
                    # Optional well ID selection
                    if "well_id" in dca_df.columns:
                        well_ids = sorted(dca_df["well_id"].dropna().unique())
                        selected_id = st.selectbox("Select well ID (if applicable)", options=well_ids)
                        dca_subset = dca_df[dca_df["well_id"] == selected_id].copy()
                    else:
                        dca_subset = dca_df.copy()
                    # Clean and sort data
                    dca_subset = dca_subset.dropna(subset=["date", rate_col])
                    dca_subset = dca_subset.sort_values("date")
                    if len(dca_subset) < 3:
                        st.warning("Not enough data points to fit a decline curve.")
                    else:
                        # Convert dates to time (years)
                        t = (dca_subset["date"] - dca_subset["date"].iloc[0]).dt.days / DAYS_PER_MONTH
                        y = dca_subset[rate_col].to_numpy(dtype=float)
                        # Decline model selection
                        model_choice = st.selectbox(
                            "Select decline model",
                            options=["Exponential", "Harmonic", "Hyperbolic"],
                            help="Hyperbolic fits the b‑factor; set b=0 for exponential tail."
                        )
                        b_override = None
                        if model_choice == "Hyperbolic":
                            b_override = st.number_input(
                                "Override b‑factor (0 for exponential tail)",
                                value=0.5,
                                min_value=0.0,
                                max_value=2.0,
                                step=0.05,
                            )
                        # Fit model on button click
                        if st.button("Fit decline model"):
                            try:
                                if model_choice == "Exponential":
                                    popt, _ = curve_fit(
                                        lambda t_val, qi_val, di_val: qi_val * np.exp(-di_val * t_val),
                                        t,
                                        y,
                                        p0=[max(y), 0.1],
                                    )
                                    qi_est, di_est = popt
                                    y_fit = qi_est * np.exp(-di_est * t)
                                    st.success(f"Exponential fit: qi={qi_est:.2f}, di={di_est:.4f}")
                                elif model_choice == "Harmonic":
                                    popt, _ = curve_fit(
                                        lambda t_val, qi_val, di_val: qi_val / (1 + di_val * t_val),
                                        t,
                                        y,
                                        p0=[max(y), 0.1],
                                    )
                                    qi_est, di_est = popt
                                    y_fit = qi_est / (1 + di_est * t)
                                    st.success(f"Harmonic fit: qi={qi_est:.2f}, di={di_est:.4f}")
                                else:
                                    # Define hyperbolic decline with b‑factor; b→0 => exponential
                                    def hyp(t_val, qi_val, di_val, b_val):
                                        if abs(b_val) < EPS:
                                            return qi_val * np.exp(-di_val * t_val)
                                        return qi_val / np.power(1.0 + b_val * di_val * t_val, 1.0 / b_val)
                                    if b_override is not None and b_override > 0:
                                        popt, _ = curve_fit(
                                            lambda t_val, qi_val, di_val: hyp(t_val, qi_val, di_val, b_override),
                                            t,
                                            y,
                                            p0=[max(y), 0.1],
                                        )
                                        qi_est, di_est = popt
                                        y_fit = hyp(t, qi_est, di_est, b_override)
                                        st.success(
                                            f"Hyperbolic fit (b={b_override:.2f}): qi={qi_est:.2f}, di={di_est:.4f}"
                                        )
                                    else:
                                        popt, _ = curve_fit(hyp, t, y, p0=[max(y), 0.1, 0.5])
                                        qi_est, di_est, b_est = popt
                                        y_fit = hyp(t, qi_est, di_est, b_est)
                                        st.success(
                                            f"Hyperbolic fit: qi={qi_est:.2f}, di={di_est:.4f}, b={b_est:.2f}"
                                        )
                                # Plot actual vs fitted
                                plot_df = pd.DataFrame({"Actual": y, "Fitted": y_fit})
                                plot_df["date"] = dca_subset["date"].values
                                plot_df = plot_df.set_index("date")
                                st.line_chart(plot_df)
                            except Exception as err:
                                st.error(f"Model fitting failed: {err}")
        except Exception as err:
            st.error(f"Failed to read CSV file: {err}")
    # Help and tutorial text
    st.header("Help & Tutorial")
    st.markdown(
        """
### Overview

This application consists of two primary tools for assessing oil & gas royalty assets:

**1. DCF Model** – Build a discounted cash flow forecast based on your assumptions about pricing, taxes, and well decline parameters. To use it:

- Use the **Scenario Management** sidebar to create or select scenarios.
- Edit global assumptions such as start date, forecast horizon, discount rate, pricing, taxes, and post‑production costs.
- Specify well parameters (initial rates, decline rates, b‑factor, royalty and NRI).
- Click **Run** to compute monthly cash flows and investment metrics (NPV, IRR, MOIC, payback periods).

**2. Decline Curve Analysis** – Fit empirical decline curves to production data. To use it:

- Upload a CSV containing a **date** column and at least one numeric **rate** column. Optionally include a **well_id** column if multiple wells are present.
- Select the rate column and well ID (if applicable).
- Choose a decline model (Exponential, Harmonic, or Hyperbolic).
- Fit the model to estimate initial rates, decline rates, and b‑factors, and visualize the fit.

These tools are intended to complement each other: you can use DCA to derive decline parameters and then input them into the DCF model. More detailed documentation can be added or linked here.
        """
    )

from forecast_model import (
    Well,
    build_price_deck,
    compute_cash_flows,
    compute_investment_metrics,
    DAYS_PER_MONTH,
    MAX_INITIAL_DECLINE,
    MAX_TERMINAL_DECLINE,
    EPS,
)

# ---------------------------------------------------------------------------
# Price deck caching
#
# Building a price deck from scratch can be an expensive operation if done
# repeatedly with identical parameters (e.g. when users adjust unrelated UI
# inputs). To keep the UI responsive, wrap the price deck builder in
# ``st.cache_data`` so that identical calls reuse the cached result. This
# caching is keyed off of the function arguments, so changes to any argument
# (start date, number of months, price levels, growth rates, or custom deck
# content) will produce a fresh deck.

if st is not None:
    @st.cache_data(show_spinner=False)
    def _cached_price_deck(
        *,
        start_date: pd.Timestamp,
        months: int,
        oil_start: float,
        oil_mom_growth: float,
        gas_start: float,
        gas_mom_growth: float,
        ngl_start: float,
        ngl_mom_growth: float,
        custom_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return build_price_deck(
            start_date=start_date,
            months=months,
            oil_start=oil_start,
            oil_mom_growth=oil_mom_growth,
            gas_start=gas_start,
            gas_mom_growth=gas_mom_growth,
            ngl_start=ngl_start,
            ngl_mom_growth=ngl_mom_growth,
            custom_df=custom_df,
        )


def get_default_scenario() -> Dict[str, Any]:
    """Return a default scenario with sensible starting assumptions."""
    today = datetime.date.today()
    model_start = datetime.date(today.year, today.month, 1)
    return {
        "model_start": model_start,
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
        "post_prod_cost_pct": 0.0,
        "wells": [
            Well(
                name="Well 1",
                first_prod_date=pd.to_datetime(model_start),
                qi_oil=300.0,
                qi_gas=800.0,
                qi_ngl=20.0,
                initial_decline=0.70,
                b_factor=0.5,
                terminal_decline=0.05,
                royalty_decimal=0.1875,
                nri=0.80,
            ).to_dict()
        ],
        "acquisition_cost": 0.0,
        "closing_date": model_start,
    }


def render_app() -> None:
    """Render the Streamlit application."""
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install streamlit and run this script via 'streamlit run'."
        )

    st.set_page_config(layout="wide")

    # Initialize session state for scenarios
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = {"Default Case": get_default_scenario()}
    if "active_scenario" not in st.session_state:
        st.session_state.active_scenario = "Default Case"
    if "uploaded_deck" not in st.session_state:
        st.session_state.uploaded_deck = None

    # Sidebar: scenario selection and management
    st.sidebar.header("Scenario Management")
    scenario_names = list(st.session_state.scenarios.keys())
    st.session_state.active_scenario = st.sidebar.selectbox(
        "Active Scenario",
        scenario_names,
        index=scenario_names.index(st.session_state.active_scenario),
        key="scenario_selector",
    )
    active_data = st.session_state.scenarios[st.session_state.active_scenario]

    with st.sidebar.expander("Create / Delete Scenario"):
        new_name = st.text_input("New Scenario Name")
        if st.button("Save as New Scenario"):
            if new_name and new_name not in st.session_state.scenarios:
                # Deep copy the current scenario to ensure independence
                st.session_state.scenarios[new_name] = copy.deepcopy(active_data)
                st.session_state.active_scenario = new_name
                st.experimental_rerun()
            else:
                st.warning("Please enter a unique name for the new scenario.")
        if st.button("Delete Current Scenario", type="secondary"):
            if st.session_state.active_scenario != "Default Case":
                del st.session_state.scenarios[st.session_state.active_scenario]
                st.session_state.active_scenario = "Default Case"
                st.experimental_rerun()
            else:
                st.warning("Cannot delete the Default Case.")

    st.sidebar.markdown("---")
    st.sidebar.header("Global Assumptions")
    # Update scenario fields from sidebar inputs
    active_data["model_start"] = st.sidebar.date_input(
        "Model start date",
        value=active_data["model_start"],
        help="All production and price forecasting begins on this date.",
    )
    active_data["forecast_months"] = st.sidebar.number_input(
        "Forecast horizon (months)",
        min_value=12,
        max_value=480,
        value=active_data["forecast_months"],
        step=12,
        help="Number of months to forecast (e.g., 360 for 30 years).",
    )
    active_data["discount_rate"] = (
        st.sidebar.number_input(
            "Discount rate (annual %)",
            min_value=0.0,
            max_value=100.0,
            value=active_data["discount_rate"] * 100,
            step=0.5,
        )
        / 100.0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Pricing")
    active_data["price_option"] = st.sidebar.selectbox(
        "Price deck option",
        ["Flat pricing", "Upload market strip"],
        index=["Flat pricing", "Upload market strip"].index(active_data["price_option"]),
    )
    active_data["oil_price"] = st.sidebar.number_input(
        "Oil price ($/bbl)", value=active_data["oil_price"], min_value=0.0
    )
    active_data["gas_price"] = st.sidebar.number_input(
        "Gas price ($/MMBtu)", value=active_data["gas_price"], min_value=0.0
    )
    active_data["ngl_price"] = st.sidebar.number_input(
        "NGL price ($/bbl)", value=active_data["ngl_price"], min_value=0.0
    )
    active_data["diff_oil"] = st.sidebar.number_input(
        "Oil price differential ($/bbl)", value=active_data["diff_oil"], format="%.2f"
    )
    active_data["diff_gas"] = st.sidebar.number_input(
        "Gas price differential ($/MMBtu)", value=active_data["diff_gas"], format="%.2f"
    )
    active_data["diff_ngl"] = st.sidebar.number_input(
        "NGL price differential ($/bbl)", value=active_data["diff_ngl"], format="%.2f"
    )

    st.sidebar.subheader("Taxes & Costs")
    active_data["severance_tax_oil"] = (
        st.sidebar.number_input(
            "Oil severance tax (%)",
            value=active_data["severance_tax_oil"] * 100,
            min_value=0.0,
            max_value=100.0,
        )
        / 100.0
    )
    active_data["severance_tax_gas"] = (
        st.sidebar.number_input(
            "Gas severance tax (%)",
            value=active_data["severance_tax_gas"] * 100,
            min_value=0.0,
            max_value=100.0,
        )
        / 100.0
    )
    active_data["post_prod_cost_pct"] = (
        st.sidebar.number_input(
            "Post‑production cost (%)",
            value=active_data.get("post_prod_cost_pct", active_data.get("post_prod_pct", 0.0)) * 100,
            min_value=0.0,
            max_value=50.0,
        )
        / 100.0
    )

    if active_data["price_option"] == "Upload market strip":
        uploaded = st.sidebar.file_uploader(
            "Upload CSV with columns: date, oil, gas, ngl",
            type=["csv"],
        )
        st.session_state.uploaded_deck = pd.read_csv(uploaded) if uploaded else None
    else:
        st.session_state.uploaded_deck = None

    # Main content
    st.title("Oil & Gas Royalty DCF Toolkit")
    st.info(f"**Current Scenario: {st.session_state.active_scenario}**")

    st.header("Well Parameters")
    # Number of wells
    well_count = st.number_input(
        "Number of wells", min_value=1, max_value=50, value=len(active_data["wells"]), step=1
    )
    current_wells = active_data["wells"]
    # Adjust wells list length
    if well_count > len(current_wells):
        for i in range(len(current_wells), well_count):
            default_date = pd.to_datetime(active_data["model_start"]) + pd.DateOffset(months=i)
            current_wells.append(
                Well(
                    name=f"Well {i+1}",
                    first_prod_date=default_date,
                    qi_oil=300.0,
                    qi_gas=800.0,
                    qi_ngl=20.0,
                    initial_decline=0.70,
                    b_factor=0.5,
                    terminal_decline=0.05,
                    royalty_decimal=0.1875,
                    nri=0.80,
                ).to_dict()
            )
    elif well_count < len(current_wells):
        active_data["wells"] = current_wells[:well_count]

    # Edit each well
    for i, w_dict in enumerate(active_data["wells"]):
        with st.expander(f"Well {i+1}: {w_dict['name']}"):
            # Work on a local reference to update the dict in place
            well = w_dict
            well["name"] = st.text_input("Well name/ID", value=well["name"], key=f"name_{i}")
            well_type = st.selectbox(
                "Well type",
                options=["PDP", "PUD", "Future"],
                index=["PDP", "PUD", "Future"].index(well.get("well_type", "PDP")),
                key=f"type_{i}",
            )
            well["well_type"] = well_type
            well["first_prod_date"] = st.date_input(
                "First production date",
                value=pd.to_datetime(well["first_prod_date"]).date(),
                key=f"fpd_{i}",
            ).isoformat()
            c1, c2, c3 = st.columns(3)
            well["qi_oil"] = c1.number_input(
                "Initial oil rate (bbl/d)", value=float(well["qi_oil"]), min_value=0.0, key=f"qi_oil_{i}"
            )
            well["qi_gas"] = c2.number_input(
                "Initial gas rate (Mcf/d)", value=float(well["qi_gas"]), min_value=0.0, key=f"qi_gas_{i}"
            )
            well["qi_ngl"] = c3.number_input(
                "Initial NGL rate (bbl/d)", value=float(well["qi_ngl"]), min_value=0.0, key=f"qi_ngl_{i}"
            )
            c1, c2, c3 = st.columns(3)
            well["b_factor"] = c1.number_input(
                "Arps b‑factor",
                value=float(well["b_factor"]),
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                key=f"b_{i}",
            )
            init_decl = c2.number_input(
                "Initial decline (%/yr)",
                value=float(well["initial_decline"] * 100),
                min_value=0.0,
                max_value=MAX_INITIAL_DECLINE * 100,
                key=f"di_{i}",
            ) / 100.0
            term_decl = c3.number_input(
                "Terminal decline (%/yr)",
                value=float(well["terminal_decline"] * 100),
                min_value=0.0,
                max_value=MAX_TERMINAL_DECLINE * 100,
                key=f"dt_{i}",
            ) / 100.0
            # Enforce logical constraint
            if init_decl < term_decl:
                st.warning("Initial decline should be greater than or equal to terminal decline. Adjusting.")
                term_decl = min(init_decl, MAX_TERMINAL_DECLINE)
            well["initial_decline"] = init_decl
            well["terminal_decline"] = term_decl
            c1, c2 = st.columns(2)
            well["royalty_decimal"] = c1.number_input(
                "Royalty decimal (fraction)",
                value=float(well["royalty_decimal"]),
                min_value=0.0,
                max_value=1.0,
                step=0.0001,
                format="%.4f",
                key=f"roy_{i}",
            )
            well["nri"] = c2.number_input(
                "Net revenue interest (NRI)",
                value=float(well["nri"]),
                min_value=0.0,
                max_value=1.0,
                step=0.0001,
                format="%.4f",
                key=f"nri_{i}",
            )

    # Acquisition inputs
    st.header("Acquisition / Investment")
    active_data["acquisition_cost"] = st.number_input(
        "Acquisition price (USD)", value=float(active_data["acquisition_cost"]), min_value=0.0, step=1000.0
    )
    if active_data["acquisition_cost"] > 0:
        active_data["closing_date"] = st.date_input(
            "Acquisition closing date",
            value=active_data["closing_date"],
            min_value=active_data["model_start"],
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    run_single = col1.button(f"Run: {st.session_state.active_scenario}", type="primary")
    run_all = col2.button("Run All Scenarios & Compare")

    if run_single or run_all:
        scenarios_to_run = (
            st.session_state.scenarios if run_all else {st.session_state.active_scenario: active_data}
        )
        all_results = []
        with st.spinner("Running forecasts..."):
            for name, scenario in scenarios_to_run.items():
                # Convert wells to dataclasses
                wells: List[Well] = [
                    Well.from_dict(w) if isinstance(w, dict) else w for w in scenario["wells"]
                ]
                # Price deck: use cached builder for efficiency. Unless a custom deck
                # is supplied via upload, assume flat pricing (zero month-over-month
                # growth). A custom deck will be passed through and used as-is.
                deck = _cached_price_deck(
                    start_date=pd.to_datetime(scenario["model_start"]),
                    months=scenario["forecast_months"],
                    oil_start=scenario["oil_price"],
                    oil_mom_growth=0.0,
                    gas_start=scenario["gas_price"],
                    gas_mom_growth=0.0,
                    ngl_start=scenario["ngl_price"],
                    ngl_mom_growth=0.0,
                    custom_df=st.session_state.uploaded_deck,
                )
                # Aggregate net cash flow across wells
                months = scenario["forecast_months"]
                agg_net = np.zeros(months)
                for w in wells:
                    df_cf, _ = compute_cash_flows(
                        wells=[w],
                        start_date=pd.to_datetime(scenario["model_start"]),
                        months=months,
                        price_deck=deck,
                        royalty_decimal=w.royalty_decimal,
                        nri=w.nri,
                        severance_tax_pct_oil=scenario["severance_tax_oil"],
                        severance_tax_pct_gas=scenario["severance_tax_gas"],
                        # Use oil severance tax as default for NGL if a distinct NGL rate is not provided.
                        severance_tax_pct_ngl=scenario.get("severance_tax_ngl", scenario["severance_tax_oil"]),
                        oil_diff=scenario["diff_oil"],
                        gas_diff=scenario["diff_gas"],
                        ngl_diff=scenario["diff_ngl"],
                        transport_cost=0.0,
                        post_prod_cost_pct=scenario.get(
                            "post_prod_cost_pct", scenario.get("post_prod_pct", 0.0)
                        ),
                        other_fixed_cost_per_month=0.0,
                    )
                    agg_net += df_cf["net_cash_flow"].to_numpy()
                res_df = pd.DataFrame({"date": deck["date"], "net_cash_flow": agg_net})
                metrics = compute_investment_metrics(
                    monthly_net_cash=agg_net,
                    acquisition_price=scenario["acquisition_cost"],
                    discount_rate_annual=scenario["discount_rate"],
                )
                all_results.append({"name": name, "metrics": metrics, "cf": res_df})
        # Display results
        st.header("Forecast Results")
        if run_all and len(all_results) > 1:
            st.subheader("Scenario Comparison")
            comp_rows = []
            for res in all_results:
                m = res["metrics"]
                comp_rows.append(
                    {
                        "Scenario": res["name"],
                        "NPV ($)": m["NPV"],
                        "IRR (%)": m["IRR"] * 100 if m["IRR"] is not None else None,
                        "MOIC (x)": m["MOIC"],
                        "Payback (months)": m["PaybackMonths"],
                    }
                )
            comp_df = pd.DataFrame(comp_rows).set_index("Scenario")
            st.dataframe(
                comp_df.style.format(
                    {
                        "NPV ($)": "{:,.0f}",
                        "IRR (%)": "{:.1f}",
                        "MOIC (x)": "{:.2f}",
                        "Payback (months)": "{:,.0f}",
                    }
                )
            )
        # Show the first result (for single run or as default view)
        first = all_results[0]
        st.subheader(f"Summary: {first['name']}")
        m = first["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("NPV ($)", f"{m['NPV']:,.0f}")
        c2.metric(
            "IRR",
            f"{m['IRR'] * 100:.1f}%" if m["IRR"] is not None else "N/A",
        )
        c3.metric(
            "MOIC",
            f"{m['MOIC']:.2f}x" if m["MOIC"] is not None else "N/A",
        )
        c1, c2 = st.columns(2)
        c1.metric(
            "Payback Months",
            f"{m['PaybackMonths']}" if m["PaybackMonths"] is not None else "N/A",
        )
        c2.metric(
            "Discounted Payback Months",
            f"{m['DiscountedPaybackMonths']}" if m["DiscountedPaybackMonths"] is not None else "N/A",
        )
        st.markdown("**Cash Flow Over Time**")
        st.line_chart(first["cf"].set_index("date"))
        # After displaying the forecast results, immediately show DCA and help,
        # and return to avoid rendering duplicate elements.
        render_dca_and_help()
        return

    # When no forecast has been run, provide access to the DCA and help sections.
    # These are hidden after a run is executed to avoid duplicate content since
    # the same sections are included in the results display.
    if not (run_single or run_all):
        # Render the DCA and help/tutorial sections once.
        render_dca_and_help()


if __name__ == "__main__":
    render_app()