import streamlit as st
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Nifty50 Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# Database configuration
db_config = {
    "host": "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "2JRRhPHCS6mRsGW.root",
    "password": "48euDwRpY6OmYT5A",
    "database": "nifty50_project",
    "ssl_ca": r"D:\path\to\ca.pem"
}

@st.cache_data(ttl=3600)
def load_data():
    """Load data from TiDB database"""
    try:
        connection = mysql.connector.connect(**db_config)
        query = """
        SELECT * FROM nifty50_data
        """
        df = pd.read_sql(query, connection)
        connection.close()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calculate necessary metrics for visualizations"""
    df = df.copy()
    df = df.sort_values(['Stock', 'Date']).reset_index(drop=True)
    
    if 'Daily_Return' not in df.columns:
        df['Daily_Return'] = df.groupby('Stock')['Close'].pct_change()
    
    volatility = df.groupby('Stock')['Daily_Return'].std().reset_index()
    volatility.columns = ['Stock', 'volatility']
    
    df['cumulative_return'] = df.groupby('Stock')['Daily_Return'].transform(
        lambda x: (1 + x.fillna(0)).cumprod() - 1
    )
    
    df['year'] = df['Date'].dt.year
    yearly_returns = df.groupby(['Stock', 'year'])['Daily_Return'].apply(
        lambda x: (1 + x.fillna(0)).prod() - 1
    ).reset_index()
    
    latest_year = df['year'].max()
    latest_returns = yearly_returns[yearly_returns['year'] == latest_year]
    
    if 'Sector' in df.columns:
        sector_info = df[['Stock', 'Sector']].drop_duplicates()
        latest_returns = pd.merge(latest_returns, sector_info, on='Stock')
    
    return df, volatility, latest_returns

def create_volatility_chart(volatility):
    """Create bar chart for top 10 volatile stocks with annotations."""
    top_10_volatile = volatility.nlargest(10, 'volatility')
    fig = px.bar(top_10_volatile, x='Stock', y='volatility',
                 title='Top 10 Most Volatile Stocks',
                 labels={'Stock': 'Stock Ticker', 'volatility': 'Volatility (Standard Deviation)'},
                 color='volatility',
                 color_continuous_scale='Viridis',
                 text_auto=True)
    fig.update_layout(title_x=0.5, title_font_size=20)
    fig.update_traces(textposition='inside')
    return fig

def create_cumulative_return_chart(df, top_5_stocks):
    """Create line chart for cumulative returns of top 5 performing stocks"""
    top_5_data = df[df['Stock'].isin(top_5_stocks)]
    fig = px.line(top_5_data, x='Date', y='cumulative_return', color='Stock',
                  title='Cumulative Return for Top 5 Performing Stocks',
                  labels={'Date': 'Date', 'cumulative_return': 'Cumulative Return'})
    fig.update_layout(title_x=0.5, title_font_size=20)
    return fig

def create_sector_returns_chart(latest_returns):
    """Create bar chart for average yearly return by sector with annotations."""
    if 'Sector' not in latest_returns.columns:
        return None
    sector_returns = latest_returns.groupby('Sector')['Daily_Return'].mean().reset_index()
    sector_returns.columns = ['Sector', 'avg_return']
    fig = px.bar(sector_returns, x='Sector', y='avg_return',
                 title='Average Yearly Return by Sector',
                 labels={'Sector': 'Sector', 'avg_return': 'Average Yearly Return'},
                 color='avg_return',
                 color_continuous_scale='RdYlGn',
                 text_auto=True)
    fig.update_layout(title_x=0.5, title_font_size=20)
    return fig

import plotly.express as px

import plotly.express as px

def create_correlation_heatmap(df):
    """Create heatmap for stock price correlations in blue color scale."""
    # Pivot to get stocks as columns
    pivot_df = df.pivot_table(index='Date', columns='Stock', values='Close')
    
    # Compute correlation matrix
    corr_matrix = pivot_df.corr()
    
    # Blue color scale
    blue_scale = "Blues"  # you can also try 'Blues_r' for reversed
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        title='Stock Price Correlation Heatmap',
        color_continuous_scale=blue_scale,
        aspect='auto',
        zmin=-1, zmax=1,        # keep consistent scale
        text_auto=True           # show correlation values
    )
    
    # Layout adjustments
    fig.update_layout(
        title_x=0.5, 
        title_font_size=20, 
        height=800,
        template='plotly_white'
    )
    
    return fig



def create_top_10_yearly_charts(latest_returns):
    """Create bar charts for the top 10 yearly gainers and losers with annotations."""
    top_10_green = latest_returns.nlargest(10, 'Daily_Return')
    top_10_loss = latest_returns.nsmallest(10, 'Daily_Return')

    fig_green = px.bar(top_10_green, x='Stock', y='Daily_Return',
                       title='Top 10 Green Stocks (Yearly Return)',
                       labels={'Stock': 'Stock', 'Daily_Return': 'Yearly Return'},
                       color='Daily_Return', color_continuous_scale='Greens',
                       text_auto=True)
    fig_green.update_layout(title_x=0.5)
    fig_green.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    
    fig_loss = px.bar(top_10_loss, x='Stock', y='Daily_Return',
                      title='Top 10 Loss Stocks (Yearly Return)',
                      labels={'Stock': 'Stock', 'Daily_Return': 'Yearly Return'},
                      color='Daily_Return', color_continuous_scale='Reds_r',
                      text_auto=True)
    fig_loss.update_layout(title_x=0.5)
    fig_loss.update_traces(texttemplate='%{y:.2%}', textposition='outside')
    
    return fig_green, fig_loss

def create_monthly_performance_dashboard(df):
    """Create a dashboard of monthly performance charts with annotations."""
    df['month_year'] = df['Date'].dt.to_period('M')
    monthly_returns = df.groupby(['Stock', 'month_year'])['Close'].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
    ).reset_index()
    monthly_returns.columns = ['Stock', 'month_year', 'monthly_return']

    unique_months = sorted(monthly_returns['month_year'].unique())
    
    cols = st.columns(3)
    
    for i, month in enumerate(unique_months):
        with cols[i % 3]:
            month_data = monthly_returns[monthly_returns['month_year'] == month]
            top_gainers = month_data.nlargest(5, 'monthly_return')
            top_losers = month_data.nsmallest(5, 'monthly_return')

            fig = make_subplots(rows=1, cols=2, 
                                subplot_titles=(f'Top 5 Gainers', f'Top 5 Losers'))
            
            # Add text labels for gainers
            fig.add_trace(go.Bar(x=top_gainers['Stock'], y=top_gainers['monthly_return'],
                                 marker_color='green', name='Gainers',
                                 text=top_gainers['monthly_return'].round(2),
                                 textposition='outside'), row=1, col=1)
            
            # Add text labels for losers
            fig.add_trace(go.Bar(x=top_losers['Stock'], y=top_losers['monthly_return'],
                                 marker_color='red', name='Losers',
                                 text=top_losers['monthly_return'].round(2),
                                 textposition='outside'), row=1, col=2)
                                 
            fig.update_layout(height=400, showlegend=False, title_text=f"Monthly Performance - {month}")
            st.plotly_chart(fig, use_container_width=True)

## Page 1: Main Dashboard
def show_main_dashboard():
    st.title("Nifty50 Stock Analysis Dashboard 📈")
    
    with st.spinner('Loading data from database...'):
        df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check your database connection.")
        return
    
    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    if 'Date' in df.columns:
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        start_date, end_date = st.sidebar.date_input(
            "Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    else:
        df_filtered = df.copy()

    # Stock filter
    if 'Stock' in df_filtered.columns:
        stocks = df_filtered['Stock'].unique()
        selected_stocks = st.sidebar.multiselect("Select stocks:", options=stocks, default=stocks[:5])
        df_display = df_filtered[df_filtered['Stock'].isin(selected_stocks)]
    else:
        df_display = df_filtered.copy()
        
    # --- Sidebar Data Overview ---
    st.sidebar.header("📊 Data Overview")
    st.sidebar.info(f"Total records in selected range: **{len(df_display)}**")
    st.sidebar.info(f"Number of stocks: **{df_display['Stock'].nunique()}**")
    
    st.sidebar.divider()
    if st.sidebar.button("Go to Monthly Performance"):
        st.session_state['page'] = 'monthly'
        st.balloons()
        st.snow()
        st.rerun()

    # --- Main Content ---
    
    # Calculate metrics
    df_metrics, volatility, latest_returns = calculate_metrics(df_filtered)
    
    # Key Metrics
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = df_metrics['Stock'].nunique()
        st.metric(label="Total Stocks", value=total_stocks)
    
    with col2:
        if not volatility.empty:
            avg_volatility = volatility['volatility'].mean()
            st.metric(label="Avg Volatility", value=f"{avg_volatility:.4f}")
    
    with col3:
        if not latest_returns.empty:
            avg_return = latest_returns['Daily_Return'].mean()
            # CORRECTED: The label for st.metric was not a single string
            st.metric(label="Avg Yearly Return", value=f"{avg_return:.2%}")
    
    with col4:
        if not latest_returns.empty:
            top_performer = latest_returns.nlargest(1, 'Daily_Return').iloc[0]
            st.metric(label="Top Performer", value=top_performer["Stock"], delta=f"{top_performer['Daily_Return']:.2%}")
    
    st.divider()

    st.header("Top 10 Yearly Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        if not latest_returns.empty:
            fig_green, _ = create_top_10_yearly_charts(latest_returns)
            st.plotly_chart(fig_green, use_container_width=True)
    
    with col2:
        if not latest_returns.empty:
            _, fig_loss = create_top_10_yearly_charts(latest_returns)
            st.plotly_chart(fig_loss, use_container_width=True)

    st.divider()
    
    st.header("Stock Performance and Volatility")
    col1, col2 = st.columns(2)
    
    with col1:
        if not volatility.empty:
            fig_volatility = create_volatility_chart(volatility)
            st.plotly_chart(fig_volatility, use_container_width=True)
    
    with col2:
        if not latest_returns.empty:
            top_5_stocks = latest_returns.nlargest(5, 'Daily_Return')['Stock'].tolist()
            fig_cumulative = create_cumulative_return_chart(df_metrics, top_5_stocks)
            st.plotly_chart(fig_cumulative, use_container_width=True)

    st.divider()
    
    st.header("Sector Analysis")
    if not latest_returns.empty:
        fig_sector = create_sector_returns_chart(latest_returns)
        if fig_sector:
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("Sector information not available in the dataset")

    st.divider()

    st.header("Correlation")
    fig_correlation = create_correlation_heatmap(df_metrics)
    st.plotly_chart(fig_correlation, use_container_width=True)
    
    st.divider()
    
    # Raw Data Table
    st.header("Raw Data")
    st.dataframe(df_display.tail(100), use_container_width=True)
    
    # Download button
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="nifty50_stock_data.csv",
        mime="text/csv",
    )
    
## Page 2: Monthly Performance Dashboard
def show_monthly_performance():
    st.title("Monthly Performance Dashboard 🗓️")
    st.write("This page provides a detailed monthly breakdown of the top 5 gainers and losers.")
    
    if st.button("Go to Main Dashboard"):
        st.session_state['page'] = 'main'
        st.balloons()
        st.snow()
        st.rerun()

    st.divider()

    df = load_data()
    if not df.empty:
        create_monthly_performance_dashboard(df)
    else:
        st.info("No data available to display monthly performance.")

# Main app logic to render the correct page
if st.session_state['page'] == 'main':
    show_main_dashboard()
elif st.session_state['page'] == 'monthly':
    show_monthly_performance()