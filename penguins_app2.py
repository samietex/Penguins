import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# Load data
df_penguins = pd.read_csv('penguins_cleaned.csv')

# Page configuration
st.set_page_config(page_title="Penguin Data Dashboard", layout="wide")

alt.themes.enable("dark")

# CSS Styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
    align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Penguin Data Dashboard")

    # Filters
    species = st.selectbox('Select Species', df_penguins['species'].unique())
    island = st.selectbox('Select Island', df_penguins['island'].unique())
    sex = st.selectbox('Select Sex', df_penguins['sex'].unique())

# Filter data based on selections
filtered_data = df_penguins[(df_penguins['species'] == species) & 
                            (df_penguins['island'] == island) & 
                            (df_penguins['sex'] == sex)]

# Main Panel
st.title("Penguin Data Analysis")

# Species Distribution
species_chart = alt.Chart(df_penguins).mark_bar().encode(
    x='species',
    y='count()'
)
st.altair_chart(species_chart, use_container_width=True)

# Island Distribution
island_chart = px.pie(df_penguins, names='island')
st.plotly_chart(island_chart, use_container_width=True)

# Biometric Data Analysis
st.subheader("Biometric Data Analysis")
col1, col2 = st.columns(2)
with col1:
    st.altair_chart(alt.Chart(filtered_data).mark_point().encode(x='flipper_length_mm', y='body_mass_g', color='species'))
with col2:
    st.altair_chart(alt.Chart(filtered_data).mark_point().encode(x='bill_length_mm', y='bill_depth_mm', color='species'))

# Additional Information
with st.expander("About this Dataset"):
    st.write("This dataset includes measurements for penguins.")

# Run this with `streamlit run your_script_name.py`
