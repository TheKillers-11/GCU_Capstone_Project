#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import plotly.graph_objects as go

# Hard-code early BTC data that is not widely available; see comments on each price 
early_BTC_data = {2009: {'Open':0.01,'Close':0.01,'Avg Price':0.01}, # Arbitrary value using a penny as the price since BTC was basically unpriced/miniscule in price at this time
                  2010: {'Open':0.18,'Close':0.18,'Avg Price':0.18}, # Arbitrary value using the average of the first price (0.05) and last price (0.30) of 2010
                  2011: {'Open':2.28,'Close':2.28,'Avg Price':2.28}, # Arbitrary value using the average of the first price (0.30) and last price (4.25) of 2011
                  2012: {'Open':9.09,'Close':9.09,'Avg Price':9.09}, # Arbitrary value using the average of the first price (4.72) and last price (13.45) of 2012
                  2013: {'Open':383.76,'Close':383.76,'Avg Price':383.76}, # Arbitrary value using the average of the first price (13.51) and last price (754.01) of 2013
                  2014: {'Open':618.73,'Close':618.73,'Avg Price':618.73}, # Arbitrary value using the average of the first price of 2014 (771.40) and September 16, 2014 price (466.06), which is the first day that yfinance does not have (754.01)
                 }
# Define the start and end dates for hard-to-get BTC prices
start_date = datetime(2009, 1, 3)
end_date = datetime(2014, 9, 16)

# Build the hard-to-get BTC price dataframe
records = []
for date in pd.date_range(start=start_date, end=end_date):
    # Grab the year of the current date to properly query the early_BTC_data dict
    year = date.year
    records.append({'Date':date.strftime('%Y-%m-%d %H:%M:%S'),'Open':early_BTC_data[year]['Open'],'Close':early_BTC_data[year]['Close'],'Avg Price':early_BTC_data[year]['Avg Price']})
new_records = pd.DataFrame(records)  

# Retrieving all Bitcoin prices on Yahoo finance, beginning on September 17th, 2014
BTC_Ticker = yf.Ticker('BTC-USD')
BTC_df = BTC_Ticker.history(period='max')

# Resetting the index, which is the date data by default; adding the "Date" column and data to the DataFrame
BTC_df.reset_index(inplace=True)
BTC_df.rename(columns={'index': 'Date'}, inplace=True)

# Formatting the BTC DataFrame; adding the average price column, which is used as the BTC price for an entire day in this project
BTC_df['Date'] = BTC_df['Date'].dt.strftime('%Y-%m-%d')
BTC_df.drop(['Stock Splits','Dividends','High','Low','Volume'],inplace=True,axis=1)

# join df's here
BTC_df = pd.concat([new_records,BTC_df], ignore_index=True, axis = 0)
BTC_df = BTC_df.reset_index(drop=True)

# Calculate Avg Price with the following formula; use this as the price for the entire date provided in the same entry
BTC_df['Avg Price'] = round(((BTC_df['Open'] + BTC_df['Close']) / 2),2)

# Cast the date column to a datetime date type
BTC_df['Date'] = pd.to_datetime(BTC_df['Date'])

# Strip the timestamp (time portion) from the date column of the new_records dataframe that was added 
BTC_df['Date'] = BTC_df['Date'].dt.date
lowest_date = BTC_df['Date'].min()
highest_date = BTC_df['Date'].max()

# Find any Missing dates
date_range = pd.date_range(lowest_date,highest_date)
missing_dates = date_range.difference(BTC_df['Date'])

# Loop through the missing dates, set the missing date equal to the previous row
# The yfinance library sometimes skips dates, and it seems to remedy itself a few days later
# The gaps must be filled in with some sort of data to avoid errors
for date in missing_dates:
    date = date.date()
    prev_row_index = BTC_df[BTC_df['Date'] < date].index[-1]
    prev_row_data = BTC_df.loc[prev_row_index]
    new_row = prev_row_data.copy()
    new_row['Date'] = date
    BTC_df = pd.concat([BTC_df.loc[:prev_row_index], new_row.to_frame().T, BTC_df.loc[prev_row_index+1:]], ignore_index=True)
BTC_df.reset_index(inplace=True)

# Cast the date column to a string for proper comparison with the date data from the Blockstream API
BTC_df['Date'] = BTC_df['Date'].astype(str)

# The Dash documentation says not to use global variables (if they are going to be manipulated by user action)
# If multiple sessions of the application are running, the global variable may store data from both, causing issues
# Instead, using a class seems to be the best way around this
class SessionState:
    def __init__(self):
        self.raw_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Price', 'Wallet'])
        self.filtered_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Price', 'Wallet'])

# Create an instance of the SessionState class
session_state = SessionState()

# Function to validate wallet addresses; if the address exists and has greater than 0 associated transactions,
# update Dash displays and add the wallets data to the global wallet_df dataframe 
def validate_wallet(bitcoin_address):
    # Check if the wallet address has already been entered
    unique_wallet_addresses = set(session_state.raw_all_wallet_df['Wallet'].unique())
    if bitcoin_address in unique_wallet_addresses:
        return False

    # Blockstream.info API endpoint URL
    url = f"https://blockstream.info/api/address/{bitcoin_address}/txs"

    # Storing all the transactions (with metadata) of the wallet
    transactions = []

    # Gathering all the unique transaction IDs from transactions in the wallet
    txid_dict = set()

    try:
        # Make calls of 25 (the limit of transactions returned) transactions via the Blockstream API until there are no more unique transaction IDs for the given public Bitcoin wallet
        while True:
            response = requests.get(url)
            response.raise_for_status()  # Check for any HTTP errors
            transactions_data = response.json()
            new_tx_count = 0
            last_new_txid = ''
            for transaction in transactions_data:
                txid = transaction['txid']
                if txid not in txid_dict:
                    new_tx_count+=1
                    txid_dict.add(txid)
                    transactions.append(transaction)
                    last_new_txid = txid

            if new_tx_count==0:
                break

            # Each call to the Blockstream API returns 25 transactions; if you pass the 25th transaction from a call, the next page of transactions can be retrieved
            url = f"https://blockstream.info/api/address/{bitcoin_address}/txs/chain/{last_new_txid}"

    # If there is an error retrieving the wallet's data from the API, it is an invalid wallet
    except requests.exceptions.RequestException as e:
        return False
    except ValueError as ve:
        return False
        
    # If the wallet has no transactions associated with it, mark it as invalid
    if len(transactions)==0:
        return False
    
    # Add the wallet's data to both the raw and filtered dataframes in the session_state object
    get_wallet_data(bitcoin_address,transactions)
    return True

# Create a dataframe of the transactions and add them to the session_state's raw and filtered dataframes respectively
def get_wallet_data(bitcoin_address,transactions):
    global BTC_df # Dash documentation suggests not using global variables if the variable will be changed by the user; BTC_df will never be changed
    wallet_df = pd.DataFrame(columns=['Date','Price','Amount','Value','Wallet'])
    total_balance = 0
    
    # Iterate through the transactions list backwards, which will allow the oldest transactions to be evaluated first
    for transaction in reversed(transactions):
        txid = transaction['txid']
        
        # Retrieve the timestamp of the transaction in UTC
        date = datetime.utcfromtimestamp(transaction['status']['block_time'])

        # Check if the address is in the inputs or outputs to determine sending or receiving
        for input_tx in transaction['vin']:
            if input_tx['prevout']['scriptpubkey_address'] == bitcoin_address:
                total_balance -= input_tx['prevout']['value']

        for output in transaction['vout']:
            if output['value']!=0 and output['scriptpubkey_address'] == bitcoin_address:
                total_balance += output['value']

        total_balance_in_BTC = total_balance/100000000
        
        # Grab the date portion of the datetime timestamp only as a string
        # Find the date in the BTC_df to pull the price on that date
        check_date = str(date)[:10]
        price_entry = BTC_df[BTC_df['Date'] == check_date]
        price = price_entry['Avg Price'].iloc[0] 
        
        
        current_value = round(price*total_balance_in_BTC,2) # Probably should remove pricing for the time being; recalculate at final steps of filtering is easier
        new_row = pd.DataFrame([{'Date':date,'Price':price,'Value':current_value,'Amount':total_balance_in_BTC,'Wallet':bitcoin_address}])
        
        
        wallet_df = pd.concat([wallet_df,new_row],ignore_index=True)
  
    # Format floats to 2 decimal places in the Value column
    pd.set_option('float_format', '{:f}'.format)
    wallet_df['Value'] = wallet_df['Value'].apply(lambda x: '{:.2f}'.format(x))

    # Next, the df is normalized to include all dates from Bitcoin's inception to today
    # If the date doesn't exist in the dataframe, an entry for the date is added, and it is set to the previous days latest entry
    # This is needed in order to have Bitcoin portfolio-level holdings
    # If this normalization was not done, dates that appear in one wallet but not the others would show as massive outliers in the time series and inaccurately portray Bitcoin holdings for that day
    # Convert the 'Date' column to datetime objects and define start and end dates for the loop
    wallet_df['Date'] = pd.to_datetime(wallet_df['Date'])
    start_date = datetime(2009, 1, 3)    
    end_date = datetime.today()

    # Initialize an empty DataFrame to store intermediary results
    result_df = pd.DataFrame(columns=wallet_df.columns)

    # Initialize the previous_entry Series with 0s and the bitcoin_address as values
    previous_entry = pd.Series({'Amount':0,'Price':0,'Value':0,'Wallet':bitcoin_address})
    new_entries = []
    # Loop through the date range
    while start_date <= end_date:
        date_check = start_date.date()
        # If the date does not exist in the dataframe, add an entry using the previous_entry Series (as described previously)
        if date_check not in wallet_df['Date'].dt.date.values:
            #price = BTC_df.loc[BTC_df['Date'] == str(date_check), 'Avg Price'].values[0]
            new_entries.append({'Date':start_date,
                                'Amount':previous_entry['Amount'],
                               # 'Price':price,
                                'Price':BTC_df.loc[BTC_df['Date'] == str(date_check), 'Avg Price'].values[0],
                                'Value':previous_entry['Value'],
                                'Wallet':previous_entry['Wallet']})   
        # The date exists; update the previous_entry to reflect this date's entry
        else:
            previous_entry = wallet_df[wallet_df['Date'].dt.date == date_check].iloc[-1]
        start_date += timedelta(days=1)

    result_df = pd.DataFrame(new_entries)

    # Concatenate the original DataFrame and the result DataFrame
    wallet_df = pd.concat([wallet_df, result_df],axis=0)

    # Sort the DataFrame by 'Date' in ascending order; after this, the dataframe now contains the original entries and additional entries for missing dates 
    wallet_df = wallet_df.sort_values(by='Date').reset_index(drop=True)

    # The wallet_df now contains the original entries and additional entries for any missing dates in the range of Bitcoin's existence
    # Concat the normalized wallet_df to the session_state's raw_all_wallet_df, which is stored throughout the life of the Dash session 
    session_state.raw_all_wallet_df = pd.concat([session_state.raw_all_wallet_df,wallet_df],axis=0)
    session_state.raw_all_wallet_df = session_state.raw_all_wallet_df.reset_index(drop=True)
    
    # Create a copy of the session_state object's raw_all_wallet_df for use/maniuplation
    all_wallet_df = pd.DataFrame(session_state.raw_all_wallet_df) 
    
    # Set the 'Date' column to datetime instead of strings and sort the 'Date" column in descending order
    all_wallet_df['Date'] = pd.to_datetime(all_wallet_df['Date'])
    all_wallet_df = all_wallet_df.sort_values(by='Date', ascending=False)

    # Initialize an empty DataFrame to store the filtering results (latest timestamp entry for each date of each unique wallet)
    latest_entries = pd.DataFrame(columns=all_wallet_df.columns)

    # Iterate through the sorted DataFrame and select the last entry for each combination of 'Wallet' and 'Date'
    for date, group in all_wallet_df.groupby(['Wallet', all_wallet_df['Date'].dt.date]):
        latest_entry = group.head(1)  # Select the first row, which is the latest entry
        latest_entries = pd.concat([latest_entries,latest_entry],axis=0)

    # Reset the index of the resulting DataFrame; this dataframe contains the last entry for each unqiue wallet for each date
    latest_entries = latest_entries.reset_index(drop=True)

    # Convert the date column to datetime objects for sorting / summation
    latest_entries['Date'] = latest_entries['Date'].dt.date
    
    # Sum the "amount" and "value" of all dates grouping by date and price
    session_state.filtered_all_wallet_df = latest_entries.groupby(['Date', 'Price'])[['Amount', 'Value']].sum(numeric_only=True).reset_index()
    
    # The value here is technically incorrect; value should not be summed
    # Recalculate the value by looping through the filtered_all_wallet_df and multiplying amount by price
    value_list = []
    
    # Loop through the DataFrame and calculate the 'Value' for each row; add the 'Value' column using the value_list
    for index, row in session_state.filtered_all_wallet_df.iterrows():
        amount = row['Amount']
        price = row['Price']
        value = round(amount*price,2)
        value_list.append(value)
    session_state.filtered_all_wallet_df['Value'] = value_list
    
def generate_graph(filtered_df,offset=None):  
    
#     if offset==True:
        
#     if offset==False:
        
    
    
    if len(filtered_df==0):
        fig = px.line(session_state.filtered_all_wallet_df, x='Date', y='Value', labels={'Date': 'Date', 'Value': 'Value'},
              title='Time Series of "Value"',hover_data={"Value": ":$.2f", "Date": True})
        return fig
    
    
    # Find the index of the first non-zero 'amount' value
    # Since the filtered_df has entries for all dates in Bitcoin's inception, most wallets usually do not have activity that early
    # The graph displayed will be basically unreadable in these cases; thus, it makes sense to not show any entries until the first entry with an 'Amount' value above 0
    first_nonzero_index = filtered_df['Amount'].ne(0).idxmax()

    # Create a new DataFrame starting from the first non-zero 'amount'
    filtered_df = filtered_df.iloc[first_nonzero_index:]

    # Generate the graph
    fig = px.line(filtered_df, x='Date', y='Value', labels={'Date': 'Date', 'Value': 'Value'},
              title='Time Series of "Value"',hover_data={"Value": ":$.2f", "Date": True})

    darker_gold_color = 'rgb(184,134,11)'

    # Add Bitcoin Amount as a new trace
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Amount'], mode='lines',
                             name='Amount', yaxis='y2',line=dict(color=darker_gold_color,shape='spline'),
                             #hovertemplate='Date: %{x|%Y-%m-%d}<br>Amount: %{y:.8f}'))
                             hovertemplate='Date: %{x|%b %d, %Y}<br>BTC Amount: %{y:,.8f}'))
    # Customize the hover template to show Date, Value, and Bitcoin Amount
    #fig.update_traces(hovertemplate='Date: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<br>Amount: %{y2:.8f}', selector={'name': 'Value'})
    #fig.update_traces(hovertemplate='bob: %{x|%Y-%m-%d}<br>Value: $%{y:,.2f}', selector={'name': 'Value'})

    # Customize the layout for sleek y-axis ticks
    fig.update_layout(
        yaxis=dict(
            title='USD Value',
            tickmode='linear',
            tick0=filtered_df['Value'].min(),  # Set the minimum value
            dtick=(filtered_df['Value'].max() - filtered_df['Value'].min()) / 4,  # Calculate tick interval for 5 ticks
            tickformat='$,.0f',  # Format y-axis ticks as currency with 2 decimal places
            showgrid=False,       # Show grid lines on the y-axis
            gridcolor='lightgray',  # Color of grid lines
        ),
        # yaxis=dict(
        #     title='Value',
        #     tickformat='$,.2f',  # Format y-axis ticks as currency with 2 decimal places
        #     showgrid=True,       # Show grid lines on the y-axis
        #     gridcolor='lightgray',  # Color of grid lines
        # ),
        yaxis2=dict(
            title='BTC Amount',
            overlaying='y',
            side='right',
            showgrid=False,       # Hide grid lines on the y2-axis
            tickformat=',.0f',    # Format y2-axis ticks with 8 decimal places
        ),
        xaxis=dict(
            title='Date',
            showgrid=False),
        paper_bgcolor='black',  # Set the background color to black
        plot_bgcolor='black',   # Set the plot area background color to black
        font=dict(color=darker_gold_color)
    )

    # Show the plot
    return fig

# Initialize the Dash app object
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

side_card = dbc.Card(dbc.CardBody([
    dbc.Label("Enter A Bitcoin Public Wallet Address", html_for = 'wallet_address_text_input'),
    dcc.Input(id = 'wallet_address_text_input', placeholder = 'Enter wallet address'),
    html.Button('Submit', id = 'wallet_address_submit_button', n_clicks = 0),
    html.Div([
        html.Div("Addresses added:",style={'font-weight':'bold'}),
        html.Div(id = 'wallet_addresses')
    ], style = {'height':'100%'})
]), style = {'height': '100vh'})

buttons = [
    html.Br(),
    html.Br(),
    html.Button('1D',id='1D_button', style={'border': '1px solid black'}),
    html.Button('5D',id='5D_button', style={'border': '1px solid black'}),
    html.Button('1M',id='1M_button', style={'border': '1px solid black'}),
    html.Button('3M',id='3M_button', style={'border': '1px solid black'}),
    html.Button('6M',id='6M_button', style={'border': '1px solid black'}),
    html.Button('YTD',id='YTD_button', style={'border': '1px solid black'}),
    html.Button('1Y',id='1Y_button', style={'border': '1px solid black'}),
    html.Button('5Y',id='5Y_button', style={'border': '1px solid black'}),
    html.Button('ALL',id='ALL_button', style={'border': '1px solid black'})
]

wallet_graph = dcc.Graph(id='wallet_graph')
    
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(side_card, width=3),  # Side card covering 3 columns
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(buttons), width={'size':8,'offset':1})  # Offset of 1 and width 12 for the button div
            ]),  # Remove gutters to avoid padding
            wallet_graph  # Your graph component
        ], width=9)  # Content column covering 9 columns
    ])
])
    
# app.layout = dbc.Container([
#     dbc.Row([
#         dbc.Col(side_card, width=3),  # Side card covering 3 columns
#     ]),
#     dbc.Row([
#         #dbc.Col(buttons, width = {'size':2,'offset':3}),
#         dbc.Col([
#             # Your content goes here (replace with your own components)
#             html.Div(buttons),
#             wallet_graph],
#             width=9  # Content column covering 9 columns
#         )
#     ])
# ])
                     
@app.callback(
    Output('wallet_address_text_input','value'),
    Output('wallet_addresses','children'),
    Output('wallet_graph','figure'),
    Input('wallet_address_submit_button','n_clicks'),
    State('wallet_address_text_input','value'),
    State('wallet_addresses','children')
)
def update_wallet_address_display(n_clicks,input_value,wallet_addresses):
    # Catching the callback triggered by the page's initial load
    if n_clicks==0:
        # Create an empty figure to return for display
        fig = px.line(session_state.filtered_all_wallet_df, x='Date', y='Value', labels={'Date': 'Date', 'Value': 'Value'},
              title='Time Series of "Value"',hover_data={"Value": ":$.2f", "Date": True})
        
        # Returning empty values
        return input_value,wallet_addresses,fig

    if wallet_addresses is None:
        wallet_addresses = []
    if n_clicks > 0 and input_value:
        if validate_wallet(input_value): 
            input_value = '-'+str(input_value)
        else: 
            input_value = 'INVALID-'+str(input_value)
        wallet_addresses.append(html.Div(input_value))
        input_value = ''
    fig = generate_graph(session_state.filtered_all_wallet_df)
    return input_value,wallet_addresses,fig

@app.callback(
    Output('wallet_graph','figure'),
    Input('1D_button','n_clicks'),
    Input('5D_button','n_clicks'),
    Input('1M_button','n_clicks'),
    Input('3M_button','n_clicks'),
    Input('6M_button','n_clicks'),
    Input('YTD_button','n_clicks'),
    Input('1Y_button','n_clicks'),
    Input('5Y_button','n_clicks'),
    Input('ALL_button','n_clicks'))
def time_filter_graph(clicks_1d, clicks_5d, clicks_1m, clicks_3m, clicks_6m, clicks_ytd, clicks_1y, clicks_5y, clicks_all):
    days_offset = 0
    if clicks_1d>0:
        days_offset = 1
    if clicks_5d>0:
        days_offset = 5
    if clicks_1m>0:
        # Operating under the assumption that a month-span is always 30 days (which is not true)
        days_offset = 30
    if clicks_3m>0:
        days_offset = 90
    if clicks_6m>0:
        days_offset = 180
    if clicks_ytd>0:
        # Calculate the difference in days between january 1st of the same year as the latest date in the session_state's filtered_all_wallet_df dataframe
        days_offset = 365
    if clicks_1y>0:
        days_offset = 1
    if clicks_5y>0:
        days_offset = 1825

    # Find the most recent date in the DataFrame; filter by the calculated offset
    latest_date = session_state.filtered_all_wallet_df['Date'].max()
    offset_days_back = most_recent_date - pd.DateOffset(days=days_offset)

    # Filter the DataFrame for the date range from one day back to the most recent date
    filtered_df = session_state.filtered_all_wallet_df[(session_state.filtered_all_wallet_df['date'] >= offset_days_back) & (session_state.filtered_all_wallet_df['date'] <= latest_date)]
    
    # Generate the graph including days_offset as a parameter to distinguish from the initial call in the update_wallet_address_display function
    fig = generate_graph(filtered_df,days_offset)
    return fig

if __name__ == "__main__":
    #app.run_server(debug=True,port=8051)
    app.run_server(jupyter_mode='external',port=9099)
    
# test values: 14bwkr3m8BWH8sgSXUiLVVS7CVEyHwz8sb, , jfkdlsjfkdsl, bc1q02mrh85muzdjk32sxu82022uke9qgjna6ydv05, 34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo
# large wallet: 1FWQiwK27EnGXb6BiBMRLJvunJQZZPMcGd


# In[ ]:




