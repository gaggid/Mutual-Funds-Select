import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set page configuration to wide mode
st.set_page_config(layout="wide")

# Reuse the scoring function from before
def calculate_composite_score(df):
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    def beta_score(series):
        return 1 - abs(series - 1)
    
    scores = pd.DataFrame()
    weights = {
        '1Yr': 0.15,
        '3Yr': 0.5,
        '5Yr': 0.35
    }
    
    metric_weights = {
        'alpha': 0.20,
        'sharpe': 0.20,
        'sortino': 0.15,
        'beta': 0.175,
        'treynor': 0.10,
        'std_dev': 0.175
    }
    
    for year in ['1Yr', '3Yr', '5Yr']:
        scores[f'alpha_{year}'] = normalize(df[f'Alpha \n {year}'])
        scores[f'beta_{year}'] = beta_score(df[f'Beta \n {year}'])
        scores[f'sharpe_{year}'] = normalize(df[f'Sharpe Ratio {year}'])
        scores[f'sortino_{year}'] = normalize(df[f'Sortino Ratio {year}'])
        scores[f'treynor_{year}'] = normalize(df[f'Treynor Ratio {year}'])
        scores[f'std_{year}'] = 1 - normalize(df[f'Std Dev \n {year}'])
    
    final_score = pd.Series(0, index=df.index)
    
    for metric, metric_weight in metric_weights.items():
        for year, year_weight in weights.items():
            if metric == 'alpha':
                final_score += scores[f'alpha_{year}'] * year_weight * metric_weight
            elif metric == 'sharpe':
                final_score += scores[f'sharpe_{year}'] * year_weight * metric_weight
            elif metric == 'sortino':
                final_score += scores[f'sortino_{year}'] * year_weight * metric_weight
            elif metric == 'beta':
                final_score += scores[f'beta_{year}'] * year_weight * metric_weight
            elif metric == 'treynor':
                final_score += scores[f'treynor_{year}'] * year_weight * metric_weight
            else:
                final_score += scores[f'std_{year}'] * year_weight * metric_weight
    
    return final_score * 100

def create_parallel_coordinates(df, top_10_indices):
    df_top10 = df.iloc[top_10_indices].copy()

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df_top10['Sharpe Ratio 5Yr'],
                    colorscale = 'Viridis'),
            dimensions = list([
                dict(range = [df_top10['Alpha \n 5Yr'].min(), df_top10['Alpha \n 5Yr'].max()],
                    label = 'Alpha 5Yr', values = df_top10['Alpha \n 5Yr']),
                dict(range = [df_top10['Sharpe Ratio 5Yr'].min(), df_top10['Sharpe Ratio 5Yr'].max()],
                    label = 'Sharpe 5Yr', values = df_top10['Sharpe Ratio 5Yr']),
                dict(range = [df_top10['Beta \n 5Yr'].min(), df_top10['Beta \n 5Yr'].max()],
                    label = 'Beta 5Yr', values = df_top10['Beta \n 5Yr']),
                dict(range = [df_top10['Std Dev \n 5Yr'].min(), df_top10['Std Dev \n 5Yr'].max()],
                    label = 'Std Dev 5Yr', values = df_top10['Std Dev \n 5Yr'])
            ])
        )
    )

    fig.update_layout(
        title="Multi-Metric Comparison of Top 10 Funds",
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
    )

    return fig

def create_radar_chart(df, top_10_indices):
    df_top10 = df.iloc[top_10_indices].copy()

    # Normalize the metrics for radar chart
    metrics = ['Alpha \n 5Yr', 'Sharpe Ratio 5Yr', 'Beta \n 5Yr', 'Std Dev \n 5Yr']
    df_normalized = df_top10[metrics].copy()
    for metric in metrics:
        if metric in ['Std Dev \n 5Yr']:  # Lower is better
            df_normalized[metric] = 1 - (df_normalized[metric] - df_normalized[metric].min()) / (df_normalized[metric].max() - df_normalized[metric].min())
        else:  # Higher is better
            df_normalized[metric] = (df_normalized[metric] - df_normalized[metric].min()) / (df_normalized[metric].max() - df_normalized[metric].min())

    fig = go.Figure()

    for i, fund in enumerate(df_top10.index):
        fig.add_trace(go.Scatterpolar(
            r=df_normalized.iloc[i],
            theta=metrics,
            name=df_top10['Fund Name'].iloc[i],
            fill='toself'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Fund Characteristics Comparison"
    )

    return fig

def create_correlation_heatmap(df, top_10_indices):
    df_top10 = df.iloc[top_10_indices].copy()
    metrics = ['Alpha \n 5Yr', 'Sharpe Ratio 5Yr', 'Beta \n 5Yr', 'Std Dev \n 5Yr', 'Sortino Ratio 5Yr']

    corr_matrix = df_top10[metrics].corr()

    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",
        aspect="auto"
    )

    fig.update_layout(
        title="Correlation Between Key Metrics",
        xaxis_title="Metrics",
        yaxis_title="Metrics"
    )

    return fig

def create_performance_timeline(df, top_10_indices):
    df_top10 = df.iloc[top_10_indices].copy()

    fig = go.Figure()

    periods = ['1Yr', '3Yr', '5Yr']

    for fund in df_top10['Fund Name']:
        fig.add_trace(go.Scatter(
            x=periods,
            y=[df_top10[df_top10['Fund Name']==fund][f'Sharpe Ratio {period}'].values[0] for period in periods],
            name=fund,
            mode='lines+markers'
        ))

    fig.update_layout(
        title="Performance Evolution Over Time",
        xaxis_title="Time Period",
        yaxis_title="Sharpe Ratio",
        showlegend=True
    )

    return fig

def create_comparison_charts(df, top_10_indices, top_20_indices):
    # Create a subset of data for top 20 funds
    df_top20 = df.iloc[top_20_indices].copy()
    # Add a category column
    df_top20['Category'] = np.where(df_top20.index.isin(top_10_indices), 'Top 10', 'Next 10')
    
    # 1. Performance vs Risk Plot
    fig_perf_risk = px.scatter(
        df_top20,
        x='Std Dev \n 5Yr',
        y='Sharpe Ratio 5Yr',
        color='Category',
        text='Fund Name',
        title='Performance vs Risk (5-Year)',
        labels={
            'Std Dev \n 5Yr': 'Risk (Standard Deviation)',
            'Sharpe Ratio 5Yr': 'Risk-Adjusted Returns (Sharpe Ratio)'
        }
    )
    fig_perf_risk.update_traces(textposition='top center')
    
    # 2. Alpha vs Beta Plot
    fig_alpha_beta = px.scatter(
        df_top20,
        x='Beta \n 5Yr',
        y='Alpha \n 5Yr',
        color='Category',
        text='Fund Name',
        title='Alpha vs Beta (5-Year)',
        labels={
            'Beta \n 5Yr': 'Market Sensitivity (Beta)',
            'Alpha \n 5Yr': 'Excess Returns (Alpha)'
        }
    )
    fig_alpha_beta.update_traces(textposition='top center')
    
    # 3. Consistency Plot (3Yr vs 5Yr returns)
    fig_consistency = px.scatter(
        df_top20,
        x='Sharpe Ratio 3Yr',
        y='Sharpe Ratio 5Yr',
        color='Category',
        text='Fund Name',
        title='Performance Consistency (3-Year vs 5-Year)',
        labels={
            'Sharpe Ratio 3Yr': '3-Year Risk-Adjusted Returns',
            'Sharpe Ratio 5Yr': '5-Year Risk-Adjusted Returns'
        }
    )
    fig_consistency.update_traces(textposition='top center')
    
    # 4. Quadrant Analysis Plot
    df_top20 = df.iloc[top_20_indices].copy()
    df_top20['Category'] = np.where(df_top20.index.isin(top_10_indices), 'Top 10', 'Next 10')
    
    # Calculate medians for quadrant lines
    perf_median = df_top20['Sharpe Ratio 5Yr'].median()
    risk_median = df_top20['Std Dev \n 5Yr'].median()
    
    fig_quadrant = go.Figure()
    
    # Add scatter plot
    for category, color in zip(['Top 10', 'Next 10'], ['blue', 'red']):
        mask = df_top20['Category'] == category
        fig_quadrant.add_trace(go.Scatter(
            x=df_top20[mask]['Std Dev \n 5Yr'],
            y=df_top20[mask]['Sharpe Ratio 5Yr'],
            mode='markers+text',
            name=category,
            text=df_top20[mask]['Fund Name'],
            textposition="top center",
            marker=dict(size=10, color=color),
            showlegend=True
        ))
    
    # Add quadrant lines
    fig_quadrant.add_hline(y=perf_median, line_dash="dash", line_color="gray")
    fig_quadrant.add_vline(x=risk_median, line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig_quadrant.add_annotation(x=risk_median-1, y=perf_median+0.5,
                              text="High Performance<br>Low Risk<br>(Ideal)",
                              showarrow=False, font=dict(size=12, color="green"))
    
    fig_quadrant.add_annotation(x=risk_median+1, y=perf_median+0.5,
                              text="High Performance<br>High Risk<br>(Aggressive)",
                              showarrow=False, font=dict(size=12, color="orange"))
    
    fig_quadrant.add_annotation(x=risk_median-1, y=perf_median-0.5,
                              text="Low Performance<br>Low Risk<br>(Conservative)",
                              showarrow=False, font=dict(size=12, color="blue"))
    
    fig_quadrant.add_annotation(x=risk_median+1, y=perf_median-0.5,
                              text="Low Performance<br>High Risk<br>(Suboptimal)",
                              showarrow=False, font=dict(size=12, color="red"))
    
    fig_quadrant.update_layout(
        title="Fund Positioning Matrix (5-Year)",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Performance (Sharpe Ratio)",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig_perf_risk, fig_alpha_beta, fig_consistency, fig_quadrant
def main():
    st.title("Mutual Fund Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload your mutual fund CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Convert numeric columns to float
        numeric_columns = [
            'Alpha \n 1Yr', 'Alpha \n 3Yr', 'Alpha \n 5Yr',
            'Beta \n 1Yr', 'Beta \n 3Yr', 'Beta \n 5Yr',
            'Sharpe Ratio 1Yr', 'Sharpe Ratio 3Yr', 'Sharpe Ratio 5Yr',
            'Sortino Ratio 1Yr', 'Sortino Ratio 3Yr', 'Sortino Ratio 5Yr',
            'Treynor Ratio 1Yr', 'Treynor Ratio 3Yr', 'Treynor Ratio 5Yr',
            'Std Dev \n 1Yr', 'Std Dev \n 3Yr', 'Std Dev \n 5Yr'
        ]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        scores = calculate_composite_score(df)
        top_20_indices = scores.nlargest(20).index
        top_10_indices = top_20_indices[:10]
        
        # Display top 10 funds
        st.header("Top 10 Mutual Funds")
        detailed_results = pd.DataFrame({
            'Fund Name': df.iloc[top_10_indices]['Fund Name'],
            'Overall Score': scores[top_10_indices].round(1),
            'Alpha 3Yr': df.iloc[top_10_indices]['Alpha \n 3Yr'].round(2),
            'Alpha 5Yr': df.iloc[top_10_indices]['Alpha \n 5Yr'].round(2),
            'Sharpe 3Yr': df.iloc[top_10_indices]['Sharpe Ratio 3Yr'].round(2),
            'Sharpe 5Yr': df.iloc[top_10_indices]['Sharpe Ratio 5Yr'].round(2),
            'Beta 3Yr': df.iloc[top_10_indices]['Beta \n 3Yr'].round(2),
            'Beta 5Yr': df.iloc[top_10_indices]['Beta \n 5Yr'].round(2)
        })
        st.dataframe(detailed_results, use_container_width=True)
        
        # Performance comparison charts
        st.header("Performance Analysis")
        fig_perf_risk, fig_alpha_beta, fig_consistency, fig_quadrant = create_comparison_charts(df, top_10_indices, top_20_indices)
        
        # Display quadrant analysis first
        st.subheader("Fund Positioning Analysis")
        st.plotly_chart(fig_quadrant, use_container_width=True)

        # Create two columns for other charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_perf_risk, use_container_width=True)
            st.plotly_chart(fig_consistency, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_alpha_beta, use_container_width=True)
            # Create figure with secondary y-axis
            fig_dist = make_subplots(specs=[[{"secondary_y": True}]])

            # Add histogram for scores
            fig_dist.add_trace(
            go.Histogram(
                x=scores,
                name="Score Distribution",
                nbinsx=20,
                opacity=0.75
            ),
            secondary_y=False
            )

            # Add line plot for Alpha 3Yr returns
            fig_dist.add_trace(
            go.Scatter(
                x=scores,
                y=df['Alpha \n 3Yr'],
                name="Alpha 3Yr Returns",
                mode='markers',
                marker=dict(size=6, color='red'),
            ),
            secondary_y=True
            )

            # Update layout
            fig_dist.update_layout(
            title="Distribution of Fund Scores with Alpha Returns",
            xaxis_title="Score",
            barmode='overlay'
            )

            # Update y-axes labels
            fig_dist.update_yaxes(title_text="Number of Funds", secondary_y=False)
            fig_dist.update_yaxes(title_text="Alpha 3Yr Returns (%)", secondary_y=True)

            st.plotly_chart(fig_dist, use_container_width=True)
        
        st.header("Advanced Analysis")

            # Parallel Coordinates Plot
        fig_parallel = create_parallel_coordinates(df, top_10_indices)
        st.plotly_chart(fig_parallel, use_container_width=True)

        # Create two columns for radar and heatmap
        col1, col2 = st.columns(2)

        with col1:
            # Radar Chart
            fig_radar = create_radar_chart(df, top_10_indices)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Correlation Heatmap
            fig_heatmap = create_correlation_heatmap(df, top_10_indices)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Add Performance Timeline
        st.header("Performance Timeline Analysis")
        fig_timeline = create_performance_timeline(df, top_10_indices)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Risk Metrics
        st.header("Risk Metrics")
        risk_metrics = pd.DataFrame({
            'Fund Name': df.iloc[top_10_indices]['Fund Name'],
            'Std Dev 3Yr': df.iloc[top_10_indices]['Std Dev \n 3Yr'].round(2),
            'Std Dev 5Yr': df.iloc[top_10_indices]['Std Dev \n 5Yr'].round(2),
            'Sortino 3Yr': df.iloc[top_10_indices]['Sortino Ratio 3Yr'].round(2),
            'Sortino 5Yr': df.iloc[top_10_indices]['Sortino Ratio 5Yr'].round(2)
        })
        st.dataframe(risk_metrics, use_container_width=True)
        
        # Download results
        csv = detailed_results.to_csv(index=False)
        st.download_button(
            label="Download detailed results as CSV",
            data=csv,
            file_name="mutual_fund_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()