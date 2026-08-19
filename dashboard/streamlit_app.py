
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

DATASET_LOCATION = "Bengaluru, India"

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
        "realized_booking_value",
        "potential_revenue_lost",
        "customer_rating",
        "driver_ratings",
        "avg_vtat",
        "avg_ctat",
        "ride_distance",
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

if not default_csv:
    st.error(
        "Dashboard data file could not be found."
    )
    st.stop()

csv_path = default_csv


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

drop_col = first_existing(
    df,
    ["drop_location", "Drop Location"]
)

ride_distance_col = first_existing(
    df,
    ["ride_distance", "Ride Distance"]
)

payment_method_col = first_existing(
    df,
    ["payment_method", "Payment Method"]
)

incomplete_reason_col = first_existing(
    df,
    ["incomplete_rides_reason", "Incomplete Rides Reason"]
)

realized_value_col = first_existing(
    df,
    ["realized_booking_value"]
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
    
    st.caption("Dataset")
    st.markdown(f"📍 **{DATASET_LOCATION}**")
    st.caption(f"{len(df):,} ride bookings")

    st.divider()

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
            hourly_demand = hourly_vehicle.copy()


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
# PRODUCT & VEHICLE ECONOMICS
# =========================================================
with col1:

    with st.container(border=True):

        st.subheader("Product & Vehicle Economics")

        # -----------------------------------------
        # Revenue by vehicle type
        # -----------------------------------------
        if vehicle_col and booking_value_col:

            revenue_data = filtered.loc[
                completion_mask(filtered)
            ].dropna(
                subset=[
                    vehicle_col,
                    booking_value_col
                ]
            )


            revenue_by_vehicle = (
                revenue_data
                .groupby(
                    vehicle_col,
                    as_index=False
                )[booking_value_col]
                .sum()
                .sort_values(
                    booking_value_col,
                    ascending=True
                )
            )
            
            fig = px.bar(
                revenue_by_vehicle,
                x=booking_value_col,
                y=vehicle_col,
                color=vehicle_col,
                color_discrete_map=VEHICLE_COLORS,
                orientation="h",
                title="Completed Revenue by Vehicle Type",
                labels={
                    booking_value_col: "Completed Revenue (₹)",
                    vehicle_col: ""
                },
                text_auto=".2s",
            )
            
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=revenue_by_vehicle[vehicle_col].tolist()
            )
            
            fig.update_layout(
                showlegend=False
            )
            
            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )


        else:
            st.info(
                "Vehicle or booking value "
                "column not found."
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

            demand = demand.sort_values(
                "bookings",
                ascending=True
            )

            fig = px.bar(
                demand,
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
                text_auto=True,
            )

            fig.update_traces(
                textposition="inside",
                texttemplate="%{x:,.0f}"
            )
            
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=demand[vehicle_col].tolist()
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
        # VTAT by vehicle type
        # -----------------------------------------
        if vtat_col and vehicle_col:
        
            vtat_by_vehicle = (
                filtered
                .dropna(
                    subset=[
                        vtat_col,
                        vehicle_col
                    ]
                )
                .groupby(
                    vehicle_col,
                    as_index=False
                )[vtat_col]
                .mean()
                .sort_values(
                    vtat_col,
                    ascending=False
                )
            )
        
            fig = px.bar(
                vtat_by_vehicle,
                x=vtat_col,
                y=vehicle_col,
                color=vehicle_col,
                color_discrete_map=VEHICLE_COLORS,
                orientation="h",
                title="Average VTAT by Vehicle Type",
                labels={
                    vtat_col:
                        "Average VTAT",
                    vehicle_col: ""
                },
                text_auto=".1f",
            )

            fig.update_layout(
                showlegend=False
            )

            fig.update_traces(
                textposition="inside",
                texttemplate="%{x:.1f}"
            )
        
            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )
        
        else:
            st.info(
                "VTAT or vehicle type "
                "column not found."
            )
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

        st.subheader("Revenue & Geography")

        if payment_method_col and booking_value_col:
        
            payment_data = filtered.loc[
                completion_mask(filtered)
            ].dropna(
                subset=[
                    payment_method_col,
                    booking_value_col
                ]
            )
        
            revenue_by_payment = (
                payment_data
                .groupby(
                    payment_method_col,
                    as_index=False
                )[booking_value_col]
                .sum()
                .sort_values(
                    booking_value_col,
                    ascending=False
                )
            )
        
            fig = px.bar(
                revenue_by_payment,
                x=payment_method_col,
                y=booking_value_col,
                title="Completed Revenue by Payment Method",
                labels={
                    payment_method_col:
                        "Payment Method",
                    booking_value_col:
                        "Completed Revenue (₹)"
                },
                text_auto=".2s",
            )
        
            st.plotly_chart(
                style_chart(fig),
                use_container_width=True
            )


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
                use_container_width=True,
                key="top_cancellation_reasons_chart"
            )

        else:
            st.info(
                "Cancellation reason "
                "columns not found."
            )

# =========================================================
# ROW 3 — ROUTE ANALYSIS
# =========================================================
st.markdown("---")

with st.container(border=True):

    st.subheader("Top Routes")

    if pickup_col and drop_col:

        route_data = (
            filtered
            .dropna(
                subset=[
                    pickup_col,
                    drop_col
                ]
            )
            .groupby(
                [
                    pickup_col,
                    drop_col
                ]
            )
            .size()
            .reset_index(
                name="bookings"
            )
        )

        route_data["route"] = (
            route_data[
                pickup_col
            ].astype(str)
            + " → "
            + route_data[
                drop_col
            ].astype(str)
        )

        top_routes = (
            route_data
            .nlargest(
                15,
                "bookings"
            )
            .sort_values(
                "bookings"
            )
        )

        fig = px.bar(
            top_routes,
            x="bookings",
            y="route",
            orientation="h",
            title="Most Popular Pickup → Drop-off Routes",
            labels={
                "bookings":
                    "Bookings",
                "route": ""
            },
            text_auto=True,
        )

        st.plotly_chart(
            style_chart(
                fig,
                height=520
            ),
            use_container_width=True
        )

    else:
        st.info(
            "Pickup or drop-off "
            "location column not found."
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


# =========================================================
# VEHICLE DEMAND
# =========================================================
if vehicle_col and not filtered.empty:

    vehicle_demand_summary = (
        filtered[vehicle_col]
        .value_counts()
        .rename_axis("vehicle_type")
        .reset_index(name="bookings")
    )

    summary["vehicle_demand"] = (
        vehicle_demand_summary
        .to_dict(orient="records")
    )


# =========================================================
# HOURLY DEMAND
# =========================================================
if "hour" in filtered.columns:

    hourly_demand_summary = (
        filtered
        .groupby("hour")
        .size()
        .reset_index(name="bookings")
        .sort_values("hour")
    )

    summary["hourly_demand"] = (
        hourly_demand_summary
        .to_dict(orient="records")
    )


# =========================================================
# COMPLETED REVENUE BY VEHICLE TYPE
# =========================================================
if vehicle_col and booking_value_col:

    vehicle_revenue_summary = (
        filtered.loc[
            completion_mask(filtered)
        ]
        .dropna(
            subset=[
                vehicle_col,
                booking_value_col
            ]
        )
        .groupby(
            vehicle_col,
            as_index=False
        )[booking_value_col]
        .sum()
        .rename(
            columns={
                vehicle_col: "vehicle_type",
                booking_value_col: "completed_revenue_inr"
            }
        )
        .sort_values(
            "completed_revenue_inr",
            ascending=False
        )
    )

    summary["vehicle_revenue"] = (
        vehicle_revenue_summary
        .to_dict(orient="records")
    )


# =========================================================
# AVERAGE VTAT BY VEHICLE TYPE
# =========================================================
if vehicle_col and vtat_col:

    vtat_vehicle_summary = (
        filtered
        .dropna(
            subset=[
                vehicle_col,
                vtat_col
            ]
        )
        .groupby(
            vehicle_col,
            as_index=False
        )[vtat_col]
        .mean()
        .rename(
            columns={
                vehicle_col: "vehicle_type",
                vtat_col: "avg_vtat"
            }
        )
        .sort_values(
            "avg_vtat",
            ascending=True
        )
    )

    vtat_vehicle_summary["avg_vtat"] = (
        vtat_vehicle_summary["avg_vtat"]
        .round(2)
    )

    summary["vtat_by_vehicle"] = (
        vtat_vehicle_summary
        .to_dict(orient="records")
    )


# =========================================================
# COMPLETED REVENUE BY PAYMENT METHOD
# =========================================================
if payment_method_col and booking_value_col:

    payment_revenue_summary = (
        filtered.loc[
            completion_mask(filtered)
        ]
        .dropna(
            subset=[
                payment_method_col,
                booking_value_col
            ]
        )
        .groupby(
            payment_method_col,
            as_index=False
        )[booking_value_col]
        .sum()
        .rename(
            columns={
                payment_method_col: "payment_method",
                booking_value_col: "completed_revenue_inr"
            }
        )
        .sort_values(
            "completed_revenue_inr",
            ascending=False
        )
    )

    summary["revenue_by_payment_method"] = (
        payment_revenue_summary
        .to_dict(orient="records")
    )


# =========================================================
# CANCELLATION SOURCE
# =========================================================
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

cancellation_source_summary = []


if customer_cancel_flag:

    customer_cancel_count = (
        filtered[
            customer_cancel_flag
        ]
        .notna()
        .sum()
    )

    cancellation_source_summary.append(
        {
            "source": "Customer",
            "cancellations": int(
                customer_cancel_count
            ),
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

    cancellation_source_summary.append(
        {
            "source": "Driver",
            "cancellations": int(
                driver_cancel_count
            ),
        }
    )


if cancellation_source_summary:

    total_source_cancellations = sum(
        item["cancellations"]
        for item in cancellation_source_summary
    )

    for item in cancellation_source_summary:

        item["share_pct"] = (
            round(
                item["cancellations"]
                / total_source_cancellations
                * 100,
                1
            )
            if total_source_cancellations > 0
            else 0
        )

    summary["cancellation_source"] = (
        cancellation_source_summary
    )


# =========================================================
# TOP CANCELLATION REASONS
# =========================================================
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

    customer_reasons_summary = (
        filtered[
            customer_reason_col
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    customer_reasons_summary = (
        customer_reasons_summary[
            ~customer_reasons_summary
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

    if not customer_reasons_summary.empty:

        temp = (
            customer_reasons_summary
            .value_counts()
            .reset_index()
        )

        temp.columns = [
            "reason",
            "count"
        ]

        reason_frames.append(
            temp
        )


if driver_reason_col:

    driver_reasons_summary = (
        filtered[
            driver_reason_col
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    driver_reasons_summary = (
        driver_reasons_summary[
            ~driver_reasons_summary
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

    if not driver_reasons_summary.empty:

        temp = (
            driver_reasons_summary
            .value_counts()
            .reset_index()
        )

        temp.columns = [
            "reason",
            "count"
        ]

        reason_frames.append(
            temp
        )


if reason_frames:

    cancellation_reasons_summary = (
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
    )

    summary["top_cancellation_reasons"] = (
        cancellation_reasons_summary
        .to_dict(orient="records")
    )


# =========================================================
# TOP ROUTES
# =========================================================
if pickup_col and drop_col:

    top_routes_summary = (
        filtered
        .dropna(
            subset=[
                pickup_col,
                drop_col
            ]
        )
        .groupby(
            [
                pickup_col,
                drop_col
            ]
        )
        .size()
        .reset_index(
            name="bookings"
        )
        .rename(
            columns={
                pickup_col: "pickup_location",
                drop_col: "drop_location"
            }
        )
        .sort_values(
            "bookings",
            ascending=False
        )
        .head(15)
    )

    summary["top_routes"] = (
        top_routes_summary
        .to_dict(orient="records")
    )


# =========================================================
# TOP PICKUP LOCATION
# =========================================================
if pickup_col and not filtered.empty:

    summary["top_pickup_location"] = (
        filtered[pickup_col]
        .value_counts()
        .index[0]
        if filtered[pickup_col].notna().any()
        else None
    )

def generate_ai_insight(summary):
    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not gemini_api_key:

        try:
            gemini_api_key = (
                st.secrets[
                    "GEMINI_API_KEY"
                ]
            )

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
- the most important overall demand pattern
- peak hourly demand patterns
- completion and cancellation performance
- differences in booking demand between vehicle types
- differences in completed revenue between vehicle types
- vehicle operational performance based on average VTAT
- which payment methods contribute the most completed revenue
- the main cancellation sources and cancellation reasons
- meaningful patterns in the most popular pickup-to-drop-off routes
- any notable differences or concentrations supported by the supplied values

Rules:
- Use only the supplied data.
- Do not claim causation where the data only shows an association.
- Clearly distinguish observations from possible explanations.
- Prioritize the most decision-relevant findings rather than describing every value.
- Compare vehicle types where differences are meaningful.
- Highlight unusually high or low values when supported by the supplied data.
- Do not infer surge pricing because surge-pricing data is not supplied.
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
