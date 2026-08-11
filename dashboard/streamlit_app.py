
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import json
from google import genai

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Uber Rides Operations Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b0f14 0%, #111821 100%);
            color: #f5f7fa;
        }

        [data-testid="stSidebar"] {
            background-color: #0d131b;
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        .dashboard-title {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.15rem;
        }

        .dashboard-subtitle {
            color: #9aa6b2;
            margin-bottom: 1.5rem;
        }

        .section-card {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1rem 0.3rem 1rem;
            margin-bottom: 1rem;
        }

        .section-label {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 14px;
            border-radius: 14px;
        }

        .ai-box {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 1.2rem;
            margin-top: 0.75rem;
        }

        h1, h2, h3 {
            color: #f5f7fa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA LOADING
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_CSV_PATHS = [
    BASE_DIR.parent / "data" / "clean_uber_rides.csv",  # recommended
    BASE_DIR / "clean_uber_rides.csv",                  # optional fallback
]


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Parse datetime where available.
    if "request_datetime" in df.columns:
        df["request_datetime"] = pd.to_datetime(
            df["request_datetime"], errors="coerce"
        )

    # Normalize booleans if they were imported/saved as text.
    for col in ["is_completed", "is_cancelled", "is_weekend"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )

    # Ensure important numeric columns are numeric.
    numeric_candidates = [
        "booking_value",
        "booking_value_inr",
        "booking_value_usd",
        "customer_rating",
        "driver_rating",
        "avg_vtat",
        "avg_ctat",
        "trip_distance",
        "trip_duration",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive hour if it does not already exist.
    if "hour" not in df.columns and "request_datetime" in df.columns:
        df["hour"] = df["request_datetime"].dt.hour

    return df


def find_default_csv():
    for path in DEFAULT_CSV_PATHS:
        if path.exists():
            return str(path)
    return ""


# =========================================================
# COLUMN HELPERS
# =========================================================
def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def completion_mask(df):
    if "is_completed" in df.columns:
        return df["is_completed"] == True

    status_col = first_existing(
        df,
        ["booking_status", "status", "Booking Status"]
    )
    if status_col:
        return (
            df[status_col]
            .astype(str)
            .str.lower()
            .str.contains("complete", na=False)
        )

    return pd.Series(False, index=df.index)


def cancellation_mask(df):
    if "is_cancelled" in df.columns:
        return df["is_cancelled"] == True

    status_col = first_existing(
        df,
        ["booking_status", "status", "Booking Status"]
    )
    if status_col:
        return (
            df[status_col]
            .astype(str)
            .str.lower()
            .str.contains("cancel", na=False)
        )

    return pd.Series(False, index=df.index)

# =========================================================
# CONSISTENT CHART COLORS
# =========================================================
VEHICLE_COLORS = {
    "Auto": "#2F80ED",            # blue
    "Bike": "#F2C94C",            # yellow
    "Go Mini": "#27AE60",         # green
    "Go Sedan": "#9B51E0",        # purple
    "Mini": "#56CCF2",            # light blue
    "Premier Sedan": "#F2994A",   # orange
    "Uber XL": "#EB5757",         # red
}


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="dashboard-title">Uber Rides Operations Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Explore demand, service performance, customer experience and cancellations.'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# DATA SOURCE
# =========================================================
default_csv = find_default_csv()

with st.sidebar:
    st.header("Dashboard Filters")

    if not default_csv:
        st.info(
            "Place `clean_uber_rides.csv` in the same folder as this app, "
            "or enter the path below."
        )

    csv_path = st.text_input(
        "CSV file path",
        value=default_csv or "clean_uber_rides.csv",
    )


try:
    df = load_data(csv_path)
except Exception as exc:
    st.error(f"Could not load the CSV file: {exc}")
    st.stop()


# =========================================================
# FILTER DEFINITIONS
# =========================================================
vehicle_col = first_existing(
    df,
    ["vehicle_type", "Vehicle Type"]
)

status_col = first_existing(
    df,
    ["booking_status", "status", "Booking Status"]
)

pickup_col = first_existing(
    df,
    ["pickup_location", "Pickup Location"]
)

booking_value_col = first_existing(
    df,
    ["booking_value", "booking_value_inr", "Booking Value"]
)

customer_rating_col = first_existing(
    df,
    ["customer_rating", "Customer Rating"]
)

vtat_col = first_existing(
    df,
    ["avg_vtat", "Avg VTAT"]
)

ctat_col = first_existing(
    df,
    ["avg_ctat", "Avg CTAT"]
)

# =========================================================
# SIDEBAR FILTERS
# =========================================================
with st.sidebar:

    st.header("Dashboard Filters")

    # =====================================================
    # BOOKING STATUS
    # =====================================================
    # =====================================================
    # BOOKING STATUS
    # =====================================================
    if status_col:
    
        st.subheader("Booking Status")
    
        status_options = (
            df[status_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    
        # Show Completed and Incomplete first,
        # then sort the remaining statuses alphabetically.
        status_priority = {
            "completed": 0,
            "incomplete": 1,
        }
    
        status_options = sorted(
            status_options,
            key=lambda status: (
                status_priority.get(
                    status.strip().lower(),
                    2
                ),
                status
            )
        )
    
        selected_statuses = []
    
        for status in status_options:
    
            checked = st.checkbox(
                status,
                value=True,
                key=f"status_{status}"
            )
    
            if checked:
                selected_statuses.append(status)
    
    else:
        selected_statuses = None
    
    st.divider()


    # =====================================================
    # TIME OF DAY
    # =====================================================
    st.subheader("Time of Day")

    time_periods = {
        "Morning": list(range(6, 12)),
        "Afternoon": list(range(12, 17)),
        "Evening": list(range(17, 22)),
        "Night": [22, 23, 0, 1, 2, 3, 4, 5],
    }



    selected_periods = []

    for period in time_periods:
    
        checked = st.checkbox(
            period,
            value=True,
            key=f"time_{period}"
        )
    
        if checked:
            selected_periods.append(period)


    st.divider()


    # =====================================================
    # VEHICLE TYPE
    # =====================================================
    # =====================================================
    # VEHICLE TYPE
    # =====================================================
    if vehicle_col:
        
        st.subheader("Vehicle Type")

        vehicle_options = sorted(
            df[vehicle_col]
            .dropna()
            .astype(str)
            .unique()
        )



        selected_vehicles = []

        for vehicle in vehicle_options:
        
            checked = st.checkbox(
                vehicle,
                value=True,
                key=f"vehicle_{vehicle}"
            )
        
            if checked:
                selected_vehicles.append(vehicle)            
    else:
        selected_vehicles = None


    st.divider()


    
    st.caption("All charts and KPIs update with these filters.")


# =========================================================
# APPLY FILTERS
# =========================================================

# =========================================================
# APPLY FILTERS
# =========================================================
filtered = df.copy()


# =========================================================
# TIME OF DAY FILTER
# =========================================================
if "hour" in filtered.columns:

    selected_hours = [
        hour
        for period in selected_periods
        for hour in time_periods[period]
    ]

    if selected_hours:
        filtered = filtered[
            filtered["hour"].isin(selected_hours)
        ]
    else:
        filtered = filtered.iloc[0:0]


# =========================================================
# VEHICLE TYPE FILTER
# =========================================================
if vehicle_col and selected_vehicles is not None:

    if selected_vehicles:
        filtered = filtered[
            filtered[vehicle_col]
            .astype(str)
            .isin(selected_vehicles)
        ]
    else:
        filtered = filtered.iloc[0:0]


# =========================================================
# BOOKING STATUS FILTER
# =========================================================
if status_col and selected_statuses is not None:

    if selected_statuses:
        filtered = filtered[
            filtered[status_col]
            .astype(str)
            .isin(selected_statuses)
        ]
    else:
        filtered = filtered.iloc[0:0]


# Stop if no data remains
if filtered.empty:

    st.warning(
        "No records match the selected filters."
    )

    st.stop()

# =========================================================
# KPI CALCULATIONS
# =========================================================
completed = completion_mask(filtered)
cancelled = cancellation_mask(filtered)

total_bookings = len(filtered)
completion_rate = completed.mean() * 100 if len(filtered) else 0
cancellation_rate = cancelled.mean() * 100 if len(filtered) else 0

avg_booking_value = (
    filtered.loc[completed, booking_value_col].mean()
    if booking_value_col
    else np.nan
)

avg_customer_rating = (
    filtered.loc[completed, customer_rating_col].mean()
    if customer_rating_col
    else np.nan
)


# =========================================================
# KPI ROW
# =========================================================
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Bookings", f"{total_bookings:,}")
k2.metric("Completion Rate", f"{completion_rate:.1f}%")
k3.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")

if booking_value_col and pd.notna(avg_booking_value):
    k4.metric(
    "Average Fare",
    f"₹ {avg_booking_value:,.0f}"
)
else:
    k4.metric("Average Fare", "N/A")

if customer_rating_col and pd.notna(avg_customer_rating):
    k5.metric("Customer Rating", f"{avg_customer_rating:.2f}")
else:
    k5.metric("Customer Rating", "N/A")

st.markdown("---")


# =========================================================
# PLOTLY THEME HELPER
# =========================================================
def style_chart(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe3ea"),
        legend_title_text="",
    )
    return fig


# =========================================================
# ROW 1
# =========================================================
# 

col1, col2 = st.columns(2, gap="large")

# =========================================================
# DEMAND & VALUE
# =========================================================
with col1:

    with st.container(border=True):

        st.subheader("Demand & Value")

        # =================================================
        # DATA FOR HOURLY CHARTS
        # =================================================

        # Use the fully filtered dataset for the hourly charts.
        hourly_filtered = filtered.copy()



        # =================================================
        # 1. BOOKING DEMAND BY HOUR
        # =================================================
        if "hour" in hourly_filtered.columns and vehicle_col:

            # Total bookings by hour



            # Bookings by hour and vehicle type
            hourly_vehicle = (
                hourly_filtered
                .groupby(["hour", vehicle_col])
                .size()
                .reset_index(name="bookings")
                .rename(
                    columns={
                        vehicle_col: "series"
                    }
                )
            )


            # Combine total + vehicle series
            hourly_demand = pd.concat(
                [
                    
                    hourly_vehicle
                ],
                ignore_index=True
            )


            fig = px.line(
                hourly_demand,
                x="hour",
                y="bookings",
                color="series",
                color_discrete_map=VEHICLE_COLORS,
                markers=True,
                title="Booking Demand by Hour of Day",
                labels={
                    "hour": "Hour of Day",
                    "bookings": "Bookings",
                    "series": "Vehicle Type"
                },
            )


            fig.update_xaxes(
                tickmode="linear",
                tick0=0,
                dtick=2,
                range=[0, 23]
            )


            st.plotly_chart(
                style_chart(
                    fig,
                    height=380
                ),
                use_container_width=True
            )

        else:

            st.info(
                "Hour or vehicle type column not found."
            )


        # =================================================
        # 2. COMPLETED BOOKING VALUE BY HOUR
        # =================================================
        if (
            "hour" in hourly_filtered.columns
            and vehicle_col
            and booking_value_col
        ):

            # Only completed rides should contribute
            # to realized booking value.
            hourly_completed_mask = completion_mask(
                hourly_filtered
            )

            value_data = hourly_filtered.loc[
                hourly_completed_mask
            ].copy()


            # Total completed booking value per hour


            


            # Completed booking value by vehicle type
            hourly_value_vehicle = (
                value_data
                .groupby(
                    ["hour", vehicle_col]
                )[booking_value_col]
                .sum()
                .reset_index()
                .rename(
                    columns={
                        vehicle_col: "series",
                        booking_value_col:
                            "booking_value"
                    }
                )
            )


            # Combine total + vehicle series
            hourly_value = pd.concat(
                [
                    
                    hourly_value_vehicle
                ],
                ignore_index=True
            )


            fig = px.line(
                hourly_value,
                x="hour",
                y="booking_value",
                color="series",
                color_discrete_map=VEHICLE_COLORS,
                markers=True,
                title=(
                    "Completed Booking Value "
                    "by Hour of Day"
                ),
                labels={
                    "hour": "Hour of Day",
                    "booking_value":
                        "Completed Booking Value",
                    "series": "Vehicle Type"
                },
            )


            fig.update_xaxes(
                tickmode="linear",
                tick0=0,
                dtick=2,
                range=[0, 23]
            )


            st.plotly_chart(
                style_chart(
                    fig,
                    height=380
                ),
                use_container_width=True
            )

        else:

            st.info(
                "Hour, vehicle type, or booking "
                "value column not found."
            )

# =========================================================
# DEMAND & VALUE
# =========================================================
with col1:

    with st.container(border=True):

        st.subheader("Demand & Value")

        # -----------------------------------------
        # Booking demand by vehicle type
        # -----------------------------------------
        if vehicle_col:

            demand = (
                filtered[vehicle_col]
                .value_counts()
                .rename_axis(vehicle_col)
                .reset_index(name="bookings")
            )

            fig = px.bar(
                demand.sort_values("bookings"),
                x="bookings",
                y=vehicle_col,
                color=vehicle_col,
                color_discrete_map=VEHICLE_COLORS,
                orientation="h",
                title="Booking Demand by Vehicle Type",
                labels={
                    "bookings": "Bookings",
                    vehicle_col: ""
                },
            )

            fig.update_layout(
                showlegend=False
            )
            
            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        else:
            st.info("Vehicle type column not found.")


        # -----------------------------------------
        # Completed booking value distribution
        # -----------------------------------------
        if booking_value_col:

            completed_values = (
                filtered.loc[
                    completed,
                    booking_value_col
                ]
                .dropna()
            )

            if not completed_values.empty:

                fig = px.histogram(
                    completed_values,
                    x=booking_value_col,
                    nbins=30,
                    title="Distribution of Completed Booking Value",
                    labels={
                        booking_value_col:
                        "Booking Value"
                    },
                )

                fig.update_yaxes(
                    title="Completed Rides"
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

            else:
                st.info(
                    "No completed booking values "
                    "for this selection."
                )

        else:
            st.info(
                "Booking value column not found."
            )


# =========================================================
# SERVICE PERFORMANCE
# =========================================================
with col2:

    with st.container(border=True):

        st.subheader("Service Performance")

        # -----------------------------------------
        # VTAT by booking outcome
        # -----------------------------------------
        if vtat_col and status_col:

            vtat_by_status = (
                filtered
                .dropna(
                    subset=[vtat_col, status_col]
                )
                .groupby(
                    status_col,
                    as_index=False
                )[vtat_col]
                .mean()
            )

            fig = px.bar(
                vtat_by_status,
                x=status_col,
                y=vtat_col,
                title=(
                    "Vehicle Arrival Time "
                    "by Booking Outcome"
                ),
                labels={
                    status_col:
                    "Booking Outcome",
                    vtat_col:
                    "Average VTAT",
                },
            )

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )

        else:
            st.info(
                "VTAT or booking status "
                "column not found."
            )


        # -----------------------------------------
        # CTAT KPI
        # -----------------------------------------
        if ctat_col:

            avg_ctat = (
                filtered[ctat_col]
                .dropna()
                .mean()
            )

            if pd.notna(avg_ctat):

                st.metric(
                    "Average CTAT",
                    f"{avg_ctat:.2f}"
                )


        # -----------------------------------------
        # CTAT by booking outcome
        # -----------------------------------------
        if ctat_col and status_col:

            ctat_by_status = (
                filtered
                .dropna(
                    subset=[ctat_col, status_col]
                )
                .groupby(
                    status_col,
                    as_index=False
                )[ctat_col]
                .mean()
            )

            if not ctat_by_status.empty:

                fig = px.bar(
                    ctat_by_status,
                    x=status_col,
                    y=ctat_col,
                    title=(
                        "Customer Arrival Time "
                        "by Booking Outcome"
                    ),
                    labels={
                        status_col:
                        "Booking Outcome",
                        ctat_col:
                        "Average CTAT",
                    },
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )


# =========================================================
# ROW 2
# =========================================================
col3, col4 = st.columns(2, gap="large")


# =========================================================
# CUSTOMER & LOCATION
# =========================================================
with col3:

    with st.container(border=True):

        st.subheader("Customer & Location")

        # -----------------------------------------
        # Customer rating distribution
        # -----------------------------------------
        if customer_rating_col:

            ratings = (
                filtered.loc[
                    completed,
                    customer_rating_col
                ]
                .dropna()
            )

            if not ratings.empty:

                fig = px.histogram(
                    ratings,
                    x=customer_rating_col,
                    nbins=20,
                    title=(
                        "Customer Rating Distribution "
                        "— Completed Rides"
                    ),
                    labels={
                        customer_rating_col:
                        "Customer Rating"
                    },
                )

                fig.update_yaxes(
                    title="Completed Rides"
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

            else:
                st.info(
                    "No customer ratings "
                    "for completed rides."
                )

        else:
            st.info(
                "Customer rating column not found."
            )


        # -----------------------------------------
        # Pickup location KPIs
        # -----------------------------------------
        if pickup_col:

            pickup_series = (
                filtered[pickup_col]
                .dropna()
            )

            unique_pickups = (
                pickup_series.nunique()
            )

            pickup_counts = (
                pickup_series
                .value_counts()
            )

            total_location_bookings = (
                pickup_counts.sum()
            )

            top10_bookings = (
                pickup_counts
                .head(10)
                .sum()
            )

            pickup_concentration = (
                top10_bookings
                / total_location_bookings
                * 100
                if total_location_bookings > 0
                else 0
            )

            loc_kpi1, loc_kpi2 = (
                st.columns(2)
            )

            loc_kpi1.metric(
                "Unique Pickup Locations",
                f"{unique_pickups:,}"
            )

            loc_kpi2.metric(
                "Top 10 Pickup Share",
                f"{pickup_concentration:.1f}%"
            )

            st.caption(
                "Top 10 Pickup Share shows "
                "how concentrated booking demand is "
                "across the most frequently used "
                "pickup locations."
            )

        else:
            st.info(
                "Pickup location column not found."
            )


# =========================================================
# CANCELLATION INSIGHTS
# =========================================================
with col4:

    with st.container(border=True):

        st.subheader("Cancellation Insights")

        # -----------------------------------------
        # Customer vs driver cancellations
        # -----------------------------------------
        customer_cancel_flag = first_existing(
            filtered,
            [
                "cancelled_rides_by_customer",
                "Cancelled Rides by Customer",
            ],
        )

        driver_cancel_flag = first_existing(
            filtered,
            [
                "cancelled_rides_by_driver",
                "Cancelled Rides by Driver",
            ],
        )

        cancel_summary = []

        if customer_cancel_flag:

            customer_cancel_count = (
                filtered[
                    customer_cancel_flag
                ]
                .notna()
                .sum()
            )

            cancel_summary.append(
                {
                    "Cancellation Source":
                        "Customer",
                    "Cancellations":
                        customer_cancel_count,
                }
            )

        if driver_cancel_flag:

            driver_cancel_count = (
                filtered[
                    driver_cancel_flag
                ]
                .notna()
                .sum()
            )

            cancel_summary.append(
                {
                    "Cancellation Source":
                        "Driver",
                    "Cancellations":
                        driver_cancel_count,
                }
            )


        if cancel_summary:

            cancel_source_df = (
                pd.DataFrame(cancel_summary)
            )

            total_source_cancels = (
                cancel_source_df[
                    "Cancellations"
                ]
                .sum()
            )

            if total_source_cancels > 0:

                cancel_source_df[
                    "Share"
                ] = (
                    cancel_source_df[
                        "Cancellations"
                    ]
                    / total_source_cancels
                    * 100
                )

                fig = px.bar(
                    cancel_source_df,
                    x="Cancellation Source",
                    y="Share",
                    title=(
                        "Customer vs Driver "
                        "Cancellation Share"
                    ),
                    labels={
                        "Share":
                        "Share of Cancellations (%)",
                        "Cancellation Source":
                        ""
                    },
                    text_auto=".1f",
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True
                )

        else:
            st.info(
                "Customer / driver cancellation "
                "columns not found."
            )


        # -----------------------------------------
        # Cancellation reasons
        # -----------------------------------------
        customer_reason_col = first_existing(
            filtered,
            [
                "reason_for_cancelling_by_customer",
                "customer_cancellation_reason",
                "Reason for cancelling by Customer",
            ],
        )

        driver_reason_col = first_existing(
            filtered,
            [
                "driver_cancellation_reason",
                "Driver Cancellation Reason",
            ],
        )


        reason_frames = []


        if customer_reason_col:

            customer_reasons = (
                filtered[
                    customer_reason_col
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            customer_reasons = (
                customer_reasons[
                    ~customer_reasons
                    .str.lower()
                    .isin(
                        [
                            "",
                            "nan",
                            "none"
                        ]
                    )
                ]
            )

            if not customer_reasons.empty:

                temp = (
                    customer_reasons
                    .value_counts()
                    .reset_index()
                )

                temp.columns = [
                    "reason",
                    "count"
                ]

                temp["source"] = (
                    "Customer"
                )

                reason_frames.append(
                    temp
                )


        if driver_reason_col:

            driver_reasons = (
                filtered[
                    driver_reason_col
                ]
                .dropna()
                .astype(str)
                .str.strip()
            )

            driver_reasons = (
                driver_reasons[
                    ~driver_reasons
                    .str.lower()
                    .isin(
                        [
                            "",
                            "nan",
                            "none"
                        ]
                    )
                ]
            )

            if not driver_reasons.empty:

                temp = (
                    driver_reasons
                    .value_counts()
                    .reset_index()
                )

                temp.columns = [
                    "reason",
                    "count"
                ]

                temp["source"] = (
                    "Driver"
                )

                reason_frames.append(
                    temp
                )


        if reason_frames:

            reasons = (
                pd.concat(
                    reason_frames,
                    ignore_index=True
                )
                .groupby(
                    "reason",
                    as_index=False
                )["count"]
                .sum()
                .nlargest(
                    10,
                    "count"
                )
                .sort_values(
                    "count"
                )
            )

            fig = px.bar(
                reasons,
                x="count",
                y="reason",
                orientation="h",
                title="Top Cancellation Reasons",
                labels={
                    "count":
                    "Cancellations",
                    "reason":
                    ""
                },
            )

            st.plotly_chart(
                style_chart(
                    fig,
                    height=380
                ),
                use_container_width=True
            )

        else:
            st.info(
                "Cancellation reason "
                "columns not found."
            )

# =========================================================
# AI-READY INSIGHT SECTION
# =========================================================
st.markdown("---")
st.subheader(
    "AI Business Interpretation"
)

summary = {
    "filters": {
        "booking_status": selected_statuses,
        "time_of_day": selected_periods,
        "vehicle_types": selected_vehicles,
    },

    "kpis": {
        "bookings": total_bookings,
        "completion_rate_pct": round(completion_rate, 1),
        "cancellation_rate_pct": round(cancellation_rate, 1),

        "avg_completed_booking_value_inr": (
            round(float(avg_booking_value), 2)
            if pd.notna(avg_booking_value)
            else None
        ),

        "avg_customer_rating": (
            round(float(avg_customer_rating), 2)
            if pd.notna(avg_customer_rating)
            else None
        ),
    },
}

# For vehicle demand
#first version

# improved version
if vehicle_col and not filtered.empty:

    vehicle_demand = (
        filtered[vehicle_col]
        .value_counts()
        .rename_axis("vehicle_type")
        .reset_index(name="bookings")
    )

    summary["vehicle_demand"] = (
        vehicle_demand
        .to_dict(orient="records")
    )
    
# For hourly demand
if "hour" in filtered.columns:

    hourly_demand_summary = (
        filtered
        .groupby("hour")
        .size()
        .reset_index(name="bookings")
    )

    summary["hourly_demand"] = (
        hourly_demand_summary
        .to_dict(orient="records")
    )
#

if pickup_col and not filtered.empty:
    summary["top_pickup_location"] = (
        filtered[pickup_col].value_counts().index[0]
        if filtered[pickup_col].notna().any()
        else None
    )

if vtat_col:

    completed_vtat = (
        filtered.loc[completed, vtat_col]
        .mean()
    )

    cancelled_vtat = (
        filtered.loc[cancelled, vtat_col]
        .mean()
    )

    summary["service_performance"] = {
        "avg_vtat_completed": (
            round(float(completed_vtat), 2)
            if pd.notna(completed_vtat)
            else None
        ),

        "avg_vtat_cancelled": (
            round(float(cancelled_vtat), 2)
            if pd.notna(cancelled_vtat)
            else None
        ),
    }

def generate_ai_insight(summary):
        
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        try:
            gemini_api_key = st.secrets["GEMINI_API_KEY"]
        except FileNotFoundError:
            gemini_api_key = None
    
    if not gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in environment variables "
            "or Streamlit secrets."
        )
    
    client = genai.Client(
        api_key=gemini_api_key
    )

    prompt = f"""
You are a business data analyst interpreting an Uber rides
operations dashboard.

The following data was calculated in Python from the currently
selected dashboard filters.

Do not recalculate the metrics.
Do not invent values that are not provided.

DATA:
{json.dumps(summary, indent=2)}

Write a concise business interpretation of the dashboard.

Focus on:
- the most important demand pattern
- operational performance
- completion and cancellation performance
- meaningful differences between vehicle types, if supported
- unusual hourly demand patterns, if visible in the supplied values

Rules:
- Use only the supplied data.
- Do not claim causation where the data only shows an association.
- Clearly distinguish observations from possible explanations.
- Avoid generic statements.
- Do not mention that you are an AI.
- Keep the response concise and suitable for an executive dashboard.
- End with one practical business recommendation.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text

# =========================================================
# CURRENT FILTER SNAPSHOT
# =========================================================
current_filters = {
    "booking_status": sorted(selected_statuses),
    "time_of_day": sorted(selected_periods),
    "vehicle_types": sorted(selected_vehicles),
}


# =========================================================
# HUMAN-READABLE FILTER SUMMARY
# =========================================================

# Booking status
if set(selected_statuses) == set(status_options):
    status_text = "All booking statuses"
else:
    status_text = ", ".join(selected_statuses)


# Time of day
if set(selected_periods) == set(time_periods.keys()):
    time_text = "All times of day"
else:
    time_text = ", ".join(selected_periods)


# Vehicle type
if set(selected_vehicles) == set(vehicle_options):
    vehicle_text = "All vehicle types"
else:
    vehicle_text = ", ".join(selected_vehicles)


filter_summary = (
    f"**Based on:** "
    f"{status_text} · "
    f"{time_text} · "
    f"{vehicle_text}"
)


# =========================================================
# DETECT FILTER CHANGES
# =========================================================

# If an interpretation has already been generated and
# the filters subsequently change, mark it as stale.
if (
    "ai_insight" in st.session_state
    and "ai_filters" in st.session_state
    and current_filters != st.session_state["ai_filters"]
):
    st.session_state["ai_is_stale"] = True


# =========================================================
# GENERATE AI INTERPRETATION
# =========================================================
if st.button(
    "✨ Generate AI Interpretation",
    type="primary"
):

    try:

        with st.spinner(
            "Interpreting the current dashboard..."
        ):

            insight = generate_ai_insight(
                summary
            )

        # Save both the interpretation and
        # the exact filters that produced it.
        st.session_state["ai_insight"] = insight
        st.session_state["ai_filters"] = current_filters

        # A freshly generated interpretation
        # is no longer stale.
        st.session_state["ai_is_stale"] = False

    except Exception as exc:

        if "503" in str(exc) or "UNAVAILABLE" in str(exc):
    
            st.warning(
                "The AI service is temporarily busy. "
                "Please try generating the interpretation again in a moment."
            )
    
        else:
    
            st.error(
                f"Could not generate AI interpretation: {exc}"
            )
        


# =========================================================
# DISPLAY AI INTERPRETATION
# =========================================================

# Only do anything if an interpretation
# has actually been generated before.
if "ai_insight" in st.session_state:

    is_stale = st.session_state.get(
        "ai_is_stale",
        False
    )

    if is_stale:

        st.info(
            "Dashboard filters have changed. "
            "Generate a new AI interpretation "
            "for the current selection."
        )

    else:

        st.markdown(filter_summary)

        with st.container(border=True):
            st.markdown(
                st.session_state["ai_insight"]
            )


# Expander
with st.expander(
    "Data sent to the AI model"
):
    st.json(summary)

st.caption(
    "Generated on demand from the metrics "
    "in the current dashboard selection."
)
