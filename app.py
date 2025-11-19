import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. CONFIGURATION
st.set_page_config(
    page_title="NexGen Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS STYLING (Theme Neutral)
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. DATA ENGINE
@st.cache_data
def load_data():
    orders = pd.read_csv("orders.csv")
    delivery = pd.read_csv("delivery_performance.csv")
    routes = pd.read_csv("routes_distance.csv")

    df = pd.merge(orders, delivery, on="Order_ID", how="left")
    df = pd.merge(df, routes, on="Order_ID", how="left")

    df["Delay_Days"] = df["Actual_Delivery_Days"] - df["Promised_Delivery_Days"]
    df["Weather_Impact"] = df["Weather_Impact"].fillna("Clear").astype(str)
    df["Traffic_Delay_Minutes"] = df["Traffic_Delay_Minutes"].fillna(0).astype(int)
    
    df["Distance_KM"] = df["Distance_KM"].fillna(1.0).replace(0, 0.01)
    df["Delivery_Cost_INR"] = df["Delivery_Cost_INR"].fillna(0)
    df["Cost_Per_KM"] = df["Delivery_Cost_INR"] / df["Distance_KM"]

    def assess_risk(row):
        if not pd.isna(row["Actual_Delivery_Days"]):
            if row["Delay_Days"] > 0: return "Late Delivery"
            return "Completed"
            
        w = row["Weather_Impact"].lower()
        if "rain" in w or "storm" in w or "fog" in w:
            return "Predicted Delay: Weather"
        
        if row["Traffic_Delay_Minutes"] > 45:
            return "Predicted Delay: Traffic"
            
        return "On Track"

    df["Status"] = df.apply(assess_risk, axis=1)
    df["Delay_Days"] = df["Delay_Days"].fillna(0)
    
    return df

try:
    df = load_data()
except Exception:
    st.stop()

# 4. SIDEBAR FILTERS
with st.sidebar:
    st.markdown("### Filters")
    
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    min_date = df["Order_Date"].min()
    max_date = df["Order_Date"].max()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    selected_segments = st.multiselect(
        "Customer Segment",
        options=df["Customer_Segment"].unique(),
        default=df["Customer_Segment"].unique()
    )
    
    selected_status = st.multiselect(
        "Delivery Status",
        options=df["Status"].unique(),
        default=df["Status"].unique()
    )

mask = (
    (df["Order_Date"] >= pd.to_datetime(date_range[0])) &
    (df["Order_Date"] <= pd.to_datetime(date_range[1])) &
    (df["Customer_Segment"].isin(selected_segments)) &
    (df["Status"].isin(selected_status))
)
filtered_df = df[mask]

# 5. MAIN DASHBOARD
st.markdown('<p class="main-header">NexGen Command Center</p>', unsafe_allow_html=True)
st.markdown("Operational Intelligence & Risk Predictive Modeling")

tab1, tab2, tab3 = st.tabs(["Executive Overview", "Root Cause Analysis", "Scenario Simulator"])

# TAB 1: OVERVIEW
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    
    total_spend = filtered_df["Delivery_Cost_INR"].sum()
    avg_delay = filtered_df["Delay_Days"].mean()
    
    risk_count = len(filtered_df[filtered_df["Status"].str.contains("Predicted|Late")])
    total_count = len(filtered_df)
    risk_ratio = (risk_count / total_count * 100) if total_count > 0 else 0
    
    c1.metric("Total Logistics Spend", f"₹{total_spend:,.0f}")
    c2.metric("Avg. Delivery Delay", f"{avg_delay:.1f} Days")
    c3.metric("Risk Exposure", f"{risk_ratio:.1f}%")
    c4.metric("Active Orders", total_count)
    
    st.subheader("Priority Watchlist (Action Required)")
    problem_orders = filtered_df[filtered_df["Status"].str.contains("Predicted|Late")].sort_values("Delay_Days", ascending=False).head(5)
    
    if not problem_orders.empty:
        st.dataframe(
            problem_orders[["Order_ID", "Customer_Segment", "Status", "Origin"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("All systems operational. No high-risk orders detected.")
        
    st.divider()
    
    st.subheader("Geographic Risk Heatmap")
    if not filtered_df.empty:
        map_data = filtered_df.groupby("Origin").agg({
            "Delay_Days": "mean",
            "Order_ID": "count",
            "Delivery_Cost_INR": "mean"
        }).reset_index()
        
        map_data["Delay_Days"] = map_data["Delay_Days"].round(1)
        
        fig_map = px.scatter(
            map_data,
            x="Origin", 
            y="Delay_Days",
            size="Order_ID",
            color="Delay_Days",
            color_continuous_scale="RdYlGn_r",
            title="Average Delay by Origin Hub (Bubble Size = Volume)",
            height=400
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No data available for selected filters.")

# TAB 2: ROOT CAUSE
with tab2:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Weather Impact Assessment")
        if not filtered_df.empty:
            fig_box = px.box(
                filtered_df,
                x="Weather_Impact",
                y="Delay_Days",
                color="Weather_Impact",
                color_discrete_sequence=px.colors.qualitative.G10,
                title="Delay Distribution vs. Weather Conditions"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
    with c2:
        st.subheader("Traffic Efficiency Correlation")
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df,
                x="Traffic_Delay_Minutes",
                y="Delivery_Cost_INR",
                color="Status",
                title="Cost Impact of Traffic Delays",
                opacity=0.7
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 3: SIMULATOR
with tab3:
    st.subheader("Cost Sensitivity Simulator")
    st.markdown("Adjust parameters to see how external factors vs. internal efficiency impact the bottom line.")
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        st.markdown("**Simulation Parameters**")
        fuel_hike = st.slider("Projected Fuel Price Increase (%)", 0, 50, 15)
        efficiency_gain = st.slider("Target Route Efficiency Gain (%)", 0, 30, 10)
    
    with col_result:
        current_cost = filtered_df["Delivery_Cost_INR"].sum()
        
        fuel_impact_amt = current_cost * (fuel_hike/100)
        intermediate_cost = current_cost + fuel_impact_amt
        efficiency_savings_amt = intermediate_cost * (efficiency_gain/100) * -1 
        
        projected_cost = intermediate_cost + efficiency_savings_amt
        variance = projected_cost - current_cost
        
        st.metric(
            "Projected Total Cost", 
            f"₹{projected_cost:,.0f}", 
            delta=f"₹{variance:,.0f} (Net Change)", 
            delta_color="inverse"
        )
        
        def fmt(x):
            return f"₹{x/1000000:.1f}M" if x > 1000000 else f"₹{x/1000:.0f}k"

        fig_waterfall = go.Figure(go.Waterfall(
            name = "Cost Impact",
            orientation = "v",
            measure = ["absolute", "relative", "relative", "total"],
            x = ["Current Cost", "Fuel Spike", "Efficiency Gain", "Final Projection"],
            y = [current_cost, fuel_impact_amt, efficiency_savings_amt, 0],
            text = [fmt(current_cost), fmt(fuel_impact_amt), fmt(efficiency_savings_amt), fmt(projected_cost)],
            textposition = "auto",
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            decreasing = {"marker":{"color":"#10B981"}},
            increasing = {"marker":{"color":"#EF4444"}},
            totals = {"marker":{"color":"#1E3A8A"}}
        ))
        
        fig_waterfall.update_layout(
            title="Financial Impact Analysis",
            waterfallgap = 0.1,
            height=400,
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)

st.divider()
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Export Analyzed Dataset", data=csv_data, file_name="NexGen_Analysis.csv", mime="text/csv")