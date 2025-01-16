Mutual Fund Analysis Dashboard
A comprehensive Streamlit-based dashboard for analyzing mutual fund performance using various metrics and interactive visualizations.

⚠️ Critical Requirement
This application specifically requires data from Trendlyne.com:

Visit Trendlyne.com
Navigate to Mutual Funds section
Go to Risk vs Reward view
Export/Download the CSV file
Use this CSV file as input for the dashboard
Note: The application will not work with any other data source or format as it's specifically designed to process Trendlyne's Risk vs Reward mutual fund data structure.

Features
Fund Scoring System: Calculates composite scores based on multiple performance metrics
Interactive Visualizations: Multiple charts and plots for in-depth analysis
Top Fund Analysis: Detailed analysis of top-performing funds
Risk-Return Analysis: Various risk and return metrics visualization
Downloadable Results: Export analysis results in CSV format
Installation
pip install -r requirements.txt
Required packages:

streamlit
pandas
numpy
plotly
Usage
Run the application:
streamlit run app.py
Upload the CSV file exported from Trendlyne.com's Risk vs Reward view
Expected Data Format (from Trendlyne.com)
The input CSV must contain these specific columns from Trendlyne:

Fund Name, Alpha 1Yr, Alpha 3Yr, Alpha 5Yr, Beta 1Yr, Beta 3Yr, Beta 5Yr,
Sharpe Ratio 1Yr, Sharpe Ratio 3Yr, Sharpe Ratio 5Yr,
Sortino Ratio 1Yr, Sortino Ratio 3Yr, Sortino Ratio 5Yr,
Treynor Ratio 1Yr, Treynor Ratio 3Yr, Treynor Ratio 5Yr,
Std Dev 1Yr, Std Dev 3Yr, Std Dev 5Yr
Dashboard Components
1. Top 10 Mutual Funds
Display of top-performing funds based on composite scoring
Key metrics including Alpha, Sharpe Ratio, and Beta
2. Performance Analysis
Fund Positioning Matrix
Performance vs Risk Plot
Alpha vs Beta Analysis
Performance Consistency Analysis
3. Advanced Analysis
Parallel Coordinates Plot
Radar Chart for Fund Characteristics
Correlation Heatmap
Performance Timeline Analysis
4. Risk Metrics
Standard Deviation Analysis
Sortino Ratio Analysis
Scoring Methodology
The composite score is calculated using:

Time period weights:

1 Year: 15%

3 Year: 50%

5 Year: 35%

Metric weights:

Alpha: 20%

Sharpe Ratio: 20%

Sortino Ratio: 15%

Beta: 17.5%

Treynor Ratio: 10%

Standard Deviation: 17.5%

Contributing
Feel free to submit issues and enhancement requests.

License
MIT License

For any issues related to data format or compatibility, please ensure you're using the correct export format from Trendlyne.com's Risk vs Reward view.