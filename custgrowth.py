import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

@st.cache_data
def growth(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

@st.cache_data
def simulate(K, r, t0, churn, acq_mean):
    months = 24
    t = np.arange(months)
    np.random.seed(42)
    active = np.maximum(growth(t, K, r, t0) + np.random.normal(0, 200, months), 0)
    new_cust = np.random.poisson(acq_mean, months)
    churned = np.cumsum(active * churn)
    params, _ = curve_fit(growth, t, active, p0=[K, r, t0])
    return active, new_cust, churned, np.cumsum(new_cust), params

st.set_page_config(page_title="Growth Simulator", layout="wide")
st.title("🚀 Startup Customer Growth Simulator")
st.markdown("**SYBSc IT Minor Project - Mathematical Modelling**")
st.markdown("**Vaswani Karishma Prakash | Roll No: 92 | SYBSc IT**")

st.sidebar.header("📊 Model Parameters")
K = st.sidebar.slider("Capacity (K)", 5000, 20000, 10000)
r = st.sidebar.slider("Growth Rate (r)", 0.1, 0.5, 0.3)
t0 = st.sidebar.slider("Inflection (t0)", 3, 12, 6)
churn = st.sidebar.slider("Churn Rate", 0.03, 0.15, 0.07)
acq = st.sidebar.slider("New Customers/mo", 300, 800, 500)

if st.sidebar.button("🔄 RUN SIMULATION"):
    active, new_cust, churned, acquired, params = simulate(K, r, t0, churn, acq)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak Active", f"{active.max():,.0f}")
    col2.metric("End Active", f"{active[-1]:,.0f}")
    col3.metric("Total Churned", f"{churned[-1]:,.0f}")
    col4.metric("Est Capacity", f"{params[0]:,.0f}")

    df = pd.DataFrame({
        'Month': range(24),
        'Active': active.round(0),
        'New': new_cust,
        'Churned': churned.round(0)
    })
    st.subheader("📈 Full Results Table")
    st.dataframe(df.tail(10))

    st.subheader("📊 Interactive Dashboard")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Growth + New Customers', 'Acquired vs Churn', 'Retention Heatmap', 'Retention Curve'),
        specs=[[{"secondary_y": True}, {}], [{}, {}]],
        vertical_spacing=0.1
    )

    months = np.arange(24)
    fig.add_trace(go.Scatter(x=months, y=active, name='Active', line=dict(color='blue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=[growth(m, *params) for m in months], name='Fitted', line=dict(color='red', dash='dash', width=3)), row=1, col=1)
    fig.add_trace(go.Bar(x=months, y=new_cust, name='New Cust', marker_color='green', opacity=0.6, yaxis='y2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=months, y=churned, name='Cumulative Churn', line=dict(color='orange', width=4)), row=1, col=2)
    fig.add_trace(go.Scatter(x=months, y=acquired, name='Total Acquired', line=dict(color='gray', width=3)), row=1, col=2)

    retention = np.linspace(1, 0.3, 12)
    heatmap_data = np.outer(np.ones(12), retention)
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=[f'M{m}' for m in range(12)],
        y=[f'C{i}' for i in range(12)],
        colorscale='Blues_r',
        zmin=0,
        zmax=1,
        text=[[f'{v:.0%}' for v in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10}
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=np.arange(12),
        y=retention * 100,
        name='Retention %',
        line=dict(color='purple', width=4),
        marker=dict(size=8)
    ), row=2, col=2)

    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Active Customers", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="New Customers", secondary_y=True, row=1, col=1)
    fig.update_layout(height=600, showlegend=True, title_text="Growth Analysis Dashboard")
    st.plotly_chart(fig, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "simulation_results.csv", "text/csv")

with st.expander("ℹ️ How to Run"):
    st.code("""
pip install streamlit numpy scipy pandas plotly
streamlit run streamlit_frontend.py
    """)
    st.info("✅ Opens in browser: localhost:8501 | Sliders update live!")

st.markdown("---")
st.caption("*Backend powered by SciPy Logistic Fitting | Graphs for SYBSc IT Report*")