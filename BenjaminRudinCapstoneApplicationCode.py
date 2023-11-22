#!/usr/bin/env python
# coding: utf-8

# In[13]:


import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
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

# Create a dataframe of the transactions and add them to the global df
def get_wallet_data(bitcoin_address,transactions):
    global BTC_df
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
        current_value = round(price*total_balance_in_BTC,2)
        new_row = pd.DataFrame([{'Date':date,'Price':price,'Value':current_value,'Amount':total_balance_in_BTC,'Wallet':bitcoin_address}])
        wallet_df = pd.concat([wallet_df,new_row],ignore_index=True)

    # Format floats to 2 decimal places in the Value column
    pd.set_option('float_format', '{:f}'.format)
    wallet_df['Value'] = wallet_df['Value'].apply(lambda x: '{:.2f}'.format(x))

    session_state.raw_all_wallet_df = pd.concat([session_state.raw_all_wallet_df,wallet_df],axis=0)
    session_state.raw_all_wallet_df = session_state.raw_all_wallet_df.reset_index(drop=True)
    
    # Create a copy of the session_state object's raw_all_wallet_df for use in this function
    all_wallet_df = pd.DataFrame(session_state.raw_all_wallet_df)
    
    # Set the 'Date' column to datetime instead of strings 
    all_wallet_df['Date'] = pd.to_datetime(all_wallet_df['Date'])

    # Sort the DataFrame by 'Date' in descending order
    all_wallet_df = all_wallet_df.sort_values(by='Date', ascending=False)

    # Initialize an empty DataFrame to store the results
    latest_entries = pd.DataFrame(columns=all_wallet_df.columns)

    # Iterate through the sorted DataFrame and select the last entry for each combination of 'Wallet' and 'Date'
    for date, group in all_wallet_df.groupby(['Wallet', all_wallet_df['Date'].dt.date]):
        latest_entry = group.head(1)  # Select the first row, which is the latest entry
        latest_entries = pd.concat([latest_entries,latest_entry],axis=0)

    # Reset the index of the resulting DataFrame; this dataframe contains the last entry for each unqiue wallet for each date
    latest_entries = latest_entries.reset_index(drop=True)
    
    # Sum each wallet for last entry of each date; gives a good high-level overview of Portfolio
    latest_entries['Date'] = latest_entries['Date'].dt.date

    # Step 2: Create a new DataFrame with summed 'Amount' and 'Value' for each unique date
    session_state.filtered_all_wallet_df = latest_entries.groupby(['Date', 'Price'])[['Amount', 'Value']].sum(numeric_only=True).reset_index()
    #session_state.filtered_all_wallet_df = latest_entries.groupby(['Date', 'Price'])[['Amount', 'Value']].sum().reset_index()
    
    value_list = []
    
    # Loop through the DataFrame and calculate the 'Value' for each row
    for index, row in session_state.filtered_all_wallet_df.iterrows():
        amount = row['Amount']
        price = row['Price']
        value = round(amount*price,2)
        value_list.append(value)

    # Add the calculated 'Value' list to the DataFrame
    session_state.filtered_all_wallet_df['Value'] = value_list
    print(session_state.filtered_all_wallet_df.tail(20))

    # Now, 'summed_entries' contains one entry for each unique date with summed 'Amount' and 'Value'
    #print(session_state.filtered_all_wallet_df)
    
    
    
    
    
    
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
    html.Button('1D', style={'border': '1px solid black'}),
    html.Button('5D', style={'border': '1px solid black'}),
    html.Button('1M', style={'border': '1px solid black'}),
    html.Button('3M', style={'border': '1px solid black'}),
    html.Button('6M', style={'border': '1px solid black'}),
    html.Button('YTD', style={'border': '1px solid black'}),
    html.Button('1Y', style={'border': '1px solid black'}),
    html.Button('5Y', style={'border': '1px solid black'}),
    html.Button('ALL', style={'border': '1px solid black'})
]

empty_graph = dcc.Graph(id='empty_graph', config={'displayModeBar': False})
    
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(side_card, width=3),  # Side card covering 3 columns
        dbc.Col([
            # Your content goes here (replace with your own components)
            html.Div(buttons),
            empty_graph],
            width=9  # Content column covering 9 columns
        )
    ])
])
# app.layout = html.Div([
#     dbc.Row([
#         dbc.Col(side_card,width=3),
#         dbc.Col(
#             dbc.Row([
#                 dbc.Col(button,width={'size':1}) for button in buttons
#             ])
#         )
#     ])
#     dbc.Row
#     dbc.Row([dbc.Col(dcc.Input(id = 'test', placeholder = 'Enter wallet address'))]),

#     dbc.Row([dbc.Col(button,width={'size':1}) for button in buttons])

                     
@app.callback(
    Output('wallet_address_text_input','value'),
    Output('wallet_addresses','children'),
    Input('wallet_address_submit_button','n_clicks'),
    State('wallet_address_text_input','value'),
    State('wallet_addresses','children')
)
def update_wallet_address_display(n_clicks,input_value,wallet_addresses):
    if wallet_addresses is None:
        wallet_addresses = []
    if n_clicks > 0 and input_value:
        if validate_wallet(input_value): # separate these functions; check if wallet is valid. If so, in the next part, call the get_wallet_data function. From there, I can get the most updated total df to return from thsi callback to the fig div
            input_value = '-'+str(input_value)
        else: 
            input_value = 'INVALID-'+str(input_value)
        wallet_addresses.append(html.Div(input_value))
        input_value = ''
    return input_value,wallet_addresses

# # Define the layout
# app.layout = dbc.Container(
#     [
#         dbc.Row(
#             [
#                 dbc.Col(left_card),  # Left card
#                 dbc.Col(
#                     [
#                         graph1,  # First graph
#                         dbc.Row(
#                             [
#                                 dbc.Col(html.Div(bitcoin_balance_box), width={"size": 4, "order": 1}),  # Bitcoin balance box
#                                 dbc.Col(html.Div(bitcoin_usd_box), width={"size": 4, "order": 2}),  # Bitcoin USD value box
#                                 dbc.Col(html.Div(calculation_method_dropdown), width={"size": 4, "order": 3}),  # Calculation method dropdown
#                             ],
#                             className="mb-4 align-items-stretch",  # Use align-items-stretch to make columns of equal height
#                         )
#                     ],
#                 )  # Graphs and additional components
#             ],
#             className="mb-4",
#         ),
#     ],
#     fluid=True,
# )
# test values: 14bwkr3m8BWH8sgSXUiLVVS7CVEyHwz8sb, 1FWQiwK27EnGXb6BiBMRLJvunJQZZPMcGd, jfkdlsjfkdsl, bc1q02mrh85muzdjk32sxu82022uke9qgjna6ydv05, 34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo

if __name__ == "__main__":
    #app.run_server(debug=True,port=8051)
    app.run_server(jupyter_mode='external')


# In[ ]:


2023-04-04 27981.680000 0.000000
1   2023-04-25 27911.240000 0.000000
2   2023-04-29 29292.530000 0.000000
3   2023-05-10 27638.200000 0.011280
4   2023-05-13 26795.920000 0.046280
5   2023-05-17 27217.140000 0.057780
6   2023-05-21 26936.120000 0.063500
7   2023-05-23 27040.840000 0.079280
8   2023-05-25 26402.830000 0.096780
9   2023-05-28 27478.400000 0.116731
10  2023-06-03 27163.730000 0.134631
11  2023-06-12 25918.390000 0.000000
12  2023-06-23 30295.930000 0.003622
13  2023-06-24 30628.720000 0.052622
14  2023-07-31 29254.210000 0.053022
15  2023-08-01 29453.300000 0.056822
16  2023-08-04 29124.240000 0.087959
17  2023-08-08 29472.760000 0.007822
18  2023-08-09 29664.090000 0.007822
19  2023-08-22 26081.200000 0.007822
20  2023-08-29 26914.940000 0.007822
21  2023-09-15 26571.260000 0.007822
22  2023-09-16 26587.240000 0.007822
23  2023-09-20 27171.120000 0.007822
24  2023-09-28 26688.680000 0.007822
25  2023-10-11 27132.700000 0.007822

                Date   Amount    Value        Price  \
0  2023-04-04 18:11:49 0.000000     0.00 27981.680000   
1  2023-04-25 19:29:38 0.000000     0.00 27911.240000   
2  2023-04-29 12:55:51 0.000000     0.00 29292.530000   
3  2023-05-10 14:46:09 0.011280   311.76 27638.200000   
4  2023-05-13 09:37:07 0.046280  1240.12 26795.920000   
5  2023-05-17 08:43:18 0.057780  1572.61 27217.140000   
6  2023-05-21 06:51:24 0.063500  1710.44 26936.120000   
7  2023-05-23 07:46:31 0.079280  2143.80 27040.840000   
8  2023-05-25 07:38:04 0.096780  2555.27 26402.830000   
9  2023-05-28 05:28:26 0.116731  3207.59 27478.400000   
10 2023-06-03 14:06:00 0.134631  3657.09 27163.730000   
11 2023-06-12 18:32:04 0.000000     0.00 25918.390000   
12 2023-06-23 10:51:55 0.003622   109.72 30295.930000   
13 2023-06-24 17:38:11 0.052622  1611.74 30628.720000   
14 2023-07-31 16:44:49 0.053022  1551.11 29254.210000   
15 2023-08-01 09:45:58 0.056822  1673.59 29453.300000   
16 2023-08-04 16:38:59 0.087959  2561.75 29124.240000   
17 2023-08-08 18:53:25 0.007822   230.53 29472.760000   
18 2023-08-09 13:18:35 0.007822   232.02 29664.090000   
19 2023-08-22 10:51:39 0.007822   204.00 26081.200000   
20 2023-08-29 18:33:11 0.007822   210.52 26914.940000   
21 2023-09-15 13:10:36 0.007822   207.83 26571.260000   
22 2023-09-16 17:51:58 0.007822   207.96 26587.240000   
23 2023-09-20 18:27:29 0.007822   212.53 27171.120000   
24 2023-09-28 12:22:46 0.007822   208.75 26688.680000   
25 2023-10-11 19:12:36 0.007822   212.22 27132.700000   
26 2023-10-11 19:45:51 0.000000     0.00 27132.700000  

