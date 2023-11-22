#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

# Style for html.buttons
button_style = {'background-color': 'black', 'color': 'rgb(184, 134, 11)', 'border': '1px solid rgb(184, 134, 11)'}

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

# Retrieve all Bitcoin prices on Yahoo finance, beginning on September 17th, 2014
BTC_Ticker = yf.Ticker('BTC-USD')
BTC_df = BTC_Ticker.history(period='max')

# Reset the index, which is the date data by default; adding the "Date" column and data to the DataFrame
BTC_df.reset_index(inplace=True)
BTC_df.rename(columns={'index': 'Date'}, inplace=True)

# Format the BTC DataFrame; add the average price column, which is used as the BTC price for an entire day in this project
BTC_df['Date'] = BTC_df['Date'].dt.strftime('%Y-%m-%d')
BTC_df.drop(['Stock Splits','Dividends','High','Low','Volume'],inplace=True,axis=1)

# Join the pricing dataframes
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

# Find any Missing dates; loop through the missing dates, set the missing date equal to the previous row
# The yfinance library sometimes skips dates, and it seems to remedy itself a few days later
# The gaps must be filled in with some sort of data to avoid errors
date_range = pd.date_range(lowest_date,highest_date)
missing_dates = date_range.difference(BTC_df['Date'])
for date in missing_dates:
    date = date.date()
    prev_row_index = BTC_df[BTC_df['Date'] < date].index[-1]
    prev_row_data = BTC_df.loc[prev_row_index]
    new_row = prev_row_data.copy()
    new_row['Date'] = date
    BTC_df = pd.concat([BTC_df.loc[:prev_row_index], new_row.to_frame().T, BTC_df.loc[prev_row_index+1:]], ignore_index=True)
BTC_df.reset_index(inplace=True)

# Cast the date column to a string for proper comparison with the Blockstream API's date data
BTC_df['Date'] = BTC_df['Date'].astype(str)

# The Dash documentation says not to use global variables if they are going to be manipulated by user action
# If multiple sessions of the application are running, the global variable may store data from both, causing issues
# Instead, using a class seems to be the best way around this
class SessionState:
    def __init__(self):
        self.raw_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Price', 'Wallet'])
        self.filtered_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Price', 'Wallet'])

# Create an instance of the SessionState class
session_state = SessionState()

# Function to validate wallet addresses
# If the address exists and has greater than 0 associated transactions, update Dash displays and add the wallets data to the session_state's wallet dfs
def validate_wallet(bitcoin_address):
    # Check if the wallet address has already been entered
    unique_wallet_addresses = set(session_state.raw_all_wallet_df['Wallet'].unique())
    if bitcoin_address in unique_wallet_addresses:
        return False

    # Blockstream.info API endpoint URL
    url = f"https://blockstream.info/api/address/{bitcoin_address}/txs"

    # Store all the transactions (with metadata) of the wallet; gather all the unique transaction IDs from transactions in the wallet
    transactions = []
    txid_set = set()

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
                if txid not in txid_set:
                    new_tx_count+=1
                    txid_set.add(txid)
                    transactions.append(transaction)
                    last_new_txid = txid

            if new_tx_count==0:
                break

            # Each call to the Blockstream API returns 25 transactions; if the 25th transaction from a call is passed, the next page of transactions is returned
            url = f"https://blockstream.info/api/address/{bitcoin_address}/txs/chain/{last_new_txid}"

    # Mark wallet as invalid if there is an error retrieving the wallet's data from the API
    except requests.exceptions.RequestException as e:
        return False
    except ValueError as ve:
        return False
        
    # Mark wallet as invalid if the wallet has no transactions associated with it
    if len(transactions)==0:
        return False
    
    # Add the wallet's data to both the raw and filtered dataframes in the session_state object; return True as the wallet is valid
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
        current_value = round(price*total_balance_in_BTC,2) # Probably should remove value for the time being; recalculate at final steps of filtering is easier
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
            new_entries.append({'Date':start_date,
                                'Amount':previous_entry['Amount'],
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
    
def generate_graph(filtered_df,button_id=None,empty_df=None):
    # Create an empty fig for initial page load if an empty_df is passed
    if len(filtered_df)==0 and len(empty_df)>0:
        filtered_df = pd.DataFrame(empty_df)
        
    # Use default title if there is no title passed
    fig_title = 'Bitcoin Amount and USD Value Time Series'
    if button_id!=None:
        button_id = button_id.split('_')[0]
        fig_title += f' {button_id}'
        
    # Specify RGB color to be used in much of the graph
    darker_gold_color = 'rgb(184,134,11)'
    
    # Generate the graph
    fig = px.line(filtered_df, x='Date', y='Value', labels={'Date': 'Date', 'Value': 'Value'},
              title=fig_title,hover_data={"Value": ":$.2f", "Date": True},color_discrete_sequence=['lightgrey'])
   
    # Add Bitcoin Amount as a new trace
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Amount'], mode='lines',
                             name='Amount', yaxis='y2',line=dict(color=darker_gold_color,shape='spline'),
                             hovertemplate='Date=%{x|%b %d, %Y}<br>Amount=%{y:,.8f}'))

    # Customize the layout for sleek y-axis ticks
    fig.update_layout(
        legend=dict(
            x=1.1, # Set the legend's x position to 1 (right)
            y=1.2, # Set the legend's y position to 1 (top)
            xanchor='right', # Specify legend's location
            yanchor='top' 
        ),
        yaxis=dict(
            title='USD Value',
            tickmode='linear',
            tick0=filtered_df['Value'].min(), # Set the lowest tick to the minimum value 
            dtick=(filtered_df['Value'].max() - filtered_df['Value'].min()) / 4, # Calculate tick interval for 5 ticks
            tickformat='$,.0f', # Format y-axis ticks as currency with 2 decimal places
            showgrid=False,       #
            gridcolor='lightgray',  
        ),
        yaxis2=dict(
            title='BTC Amount',
            overlaying='y',
            side='right',
            tickmode='linear',
            tick0 = filtered_df['Amount'].min(), # Set the lowest tick to the minimum amount
            dtick=(filtered_df['Amount'].max() - filtered_df['Amount'].min()) / 4, # Calculate tick interval for 5 ticks
            showgrid=False,       
            tickformat=',.8f', # Format y2-axis ticks with 8 decimal places
        ),
        xaxis=dict(
            title='Date',
            showgrid=False),
        paper_bgcolor='black',  
        plot_bgcolor='black',   
        font=dict(color=darker_gold_color)
    )
    
    # Add another trace so that "Value" appears on the legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Value',
                             marker=dict(color='lightgrey'), showlegend=True))
    
    # Show the plot
    return fig

# Initialize the Dash app object
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Side card object that go on the left side of the page and show the wallet addresses input
side_card = dbc.Card(dbc.CardBody([
    dbc.Label("Enter A Bitcoin Public Wallet Address", html_for = 'wallet_addresses_text_input', style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
    dcc.Input(id = 'wallet_addresses_text_input', placeholder = 'Enter wallet address', style=button_style),
    html.Button('Submit', id = 'wallet_address_submit_button', n_clicks = 0, style=button_style),
    html.Div([
        html.Div("0 Addresses Added:",id='wallet_addresses_counter_and_label',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
        html.Div(id = 'wallet_addresses'),
        html.Br(),
        html.Button('Clear Wallets', id = 'wallet_addresses_clear_button', n_clicks = 0, style=button_style)
    ], style = {'height':'100%'})
]), style = {'height': '100vh', 'background-color':'black','border':'2px solid rgb(184, 134, 11)'})

# Time filter buttons that go above the wallet_graph
buttons = [
    html.Br(),
    html.Br(),
    html.Button('1D',id='1D_button', style=button_style),
    html.Button('5D',id='5D_button', style=button_style),
    html.Button('1M',id='1M_button', style=button_style),
    html.Button('3M',id='3M_button', style=button_style),
    html.Button('6M',id='6M_button', style=button_style),
    html.Button('YTD',id='YTD_button', style=button_style),
    html.Button('1Y',id='1Y_button', style=button_style),
    html.Button('5Y',id='5Y_button', style=button_style),
    html.Button('ALL',id='ALL_button', style=button_style)
]

# Create a default empty dataframe and graph to display on the Dash application's initial load
empty_df = pd.DataFrame([{'Date':datetime.today().date(),'Amount':0,'Value':0,'Price':0}])
empty_fig = generate_graph(pd.DataFrame(),None,empty_df)

# Object that contains the wallet_graph on the page; uses the generated empty figure initially
wallet_graph = dcc.Graph(id='wallet_graph',figure=empty_fig)

# Object that contains the projection_graph on the page
projection_graph = dcc.Graph(id='projection_graph')

# Design the Dash application's layout; pay attention to width's and objects being used 
app.layout = html.Div([
    dbc.Row([
        
        # Left Side Panel for Bitcoin Address Entry
        dbc.Col(side_card, width=2), 
        dbc.Col([
            dbc.Row([
                
                # Chart Filtering Buttons
                dbc.Col(html.Div(buttons), width=5),
                dbc.Col([
                    html.Div([
                        html.Br(),
                        html.Br(),
                        
                        # Radio Items for Portfolio / Individual Wallets Time Series View
                        dbc.Row([
                            dbc.Col([
                                html.Div("View:", style={'font-weight':'bold', 'color':'rgb(184,134,11)'}),
                            ], width={'size':1, 'offset':5}),
                            dbc.Col([
                                dcc.RadioItems(
                                    id='filter_view_radio_items',
                                    options=[
                                        {'label':'Portfolio', 'value':'portfolio'},
                                        {'label':'Individual Wallets', 'value':'individual'}
                                    ],
                                    value='portfolio',
                                    labelStyle={'display':'inline', 'margin-right':'10px'},
                                    inline=True,
                                    style={'color':'rgb(184,134,11)'}
                                )
                            ], width=6)
                        ])
                    ])
                ], width = 7)
            ]),
            html.Div([wallet_graph], style={'border': '2px solid rgb(184,134,11)'}),
            html.Br(),
            
            # Portfolio metrics that go below the wallet_graph
            dbc.Row([
                dbc.Col([
                    dbc.Label("Current Bitcoin Balance:",html_for='current_bitcoin_balance',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Input(
                        id='current_bitcoin_balance',
                        value=0,
                        readOnly=True,
                        style={'width': '200px','height':'35px','text-align': 'center'}
                    )
                ], width ={'offset':1,'size':3}),
                dbc.Col([
                    dbc.Label("Current Bitcoin USD Value:",html_for='current_bitcoin_usd_value',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Input(
                        id='current_bitcoin_usd_value',
                        value=0,
                        readOnly=True,
                        style={'width': '200px','height':'35px','text-align': 'center'}
                    )
                ], width = 3),
                dbc.Col([
                    dbc.Label("Calculation Method:",html_for='read-only-input',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Dropdown(
                        id='read-only-input',
                        options=['24hr-low','24hr-high','24hr-average'],
                        value='24hr-average',
                        style={'width': '200px', 'height':'35px'}
                    )
                ], width = 3)
            ], justify = 'evenly'),  
            
            html.Br(),
            html.Br(),
            html.Br(),
            dbc.Row([
                dbc.Col([],width={'offset':1,'size':3}),
                dbc.Col([
                    dbc.Label("Price Prediction Target Year:",html_for='regression_target_year',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                    dcc.Dropdown(
                        id='regression_target_year',
                        options=[{'label': year, 'value': year} for year in [str(year) for year in range(2024, 2040)]],
                        placeholder='Select a future year.',
                        style={'width':'200px','text-align': 'center'}
                    ), 
                ], width = 3),
                dbc.Col([],width=3)
            ], justify = 'evenly'),
            
            html.Br(),
            html.Div([
                html.Div([projection_graph],style={'border': '2px solid rgb(184,134,11)'}),
                html.Br(),
                
                # Portfolio projection metrics that go below the projection graph
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Bitcoin Balance Used in Projection:",html_for='projection_bitcoin_balance',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                        dcc.Input(
                            id='projection_bitcoin_balance',
                            value=0,
                            readOnly=True,
                            style={'width':'200px','height':'35px','text-align': 'center'}
                        )
                    ], width = {'offset':1,'size':3}),
                    dbc.Col([
                        dbc.Label("Projected Bitcoin USD Value:",html_for='projected_bitcoin_usd_value',style={'font-weight':'bold','color': 'rgb(184, 134, 11)'}),
                        dcc.Input(
                            id='projected_bitcoin_usd_value',
                            value=0,
                            readOnly=True,
                            style={'width':'200px','height':'35px','text-align': 'center'}
                        )
                    ], width = 3),
                ], justify = 'evenly'),
            ], id='projection_graph_div',style={'display':'none'}),
            html.Br(),
            html.Br(),
            html.Br(),
        ], width=9)
    ])
], style={'width':'100%','background-color': 'black'})
    
@app.callback(
    Output('wallet_addresses_counter_and_label','children',allow_duplicate=True),
    Input('wallet_addresses','children'),
    prevent_initial_call=True)
def update_address_count(children):
    address_count = 0
    for child in children:
        if 'INVALID' not in child['props']['children']:
            address_count+=1
    return str(address_count)+' Addresses Added'

@app.callback(
    
     
    Output('wallet_addresses_text_input','value',allow_duplicate=True),
    Output('wallet_addresses_counter_and_label','children',allow_duplicate=True),
    Output('wallet_addresses','children',allow_duplicate=True),
    Output('filter_view_radio_items','value',allow_duplicate=True), 
    Output('wallet_graph','figure',allow_duplicate=True),
    Output('current_bitcoin_balance','value',allow_duplicate=True),
    Output('projection_bitcoin_balance','value',allow_duplicate=True),
    Output('current_bitcoin_usd_value','value',allow_duplicate=True),
#     # Level radio items
#     # Calculation Method
#     # Price Prediction Target Year... drop that whole div? 
    Input('wallet_addresses_clear_button','n_clicks'),
    prevent_initial_call=True)
def reset_application(n_clicks):
    if n_clicks>0:
        session_state.raw_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Price', 'Wallet'])
        session_state.filtered_all_wallet_df = pd.DataFrame(columns=['Date', 'Amount', 'Value', 'Price', 'Wallet'])
        return None,'0 Addresses Added:',[],'portfolio',empty_fig
    
# Callback that tracks the user wallets added and processes them (adds them to the page, generates graphs, generates session_state's dataframes for the wallet)
@app.callback(
    Output('wallet_addresses_text_input','value',allow_duplicate=True),
    Output('wallet_addresses','children',allow_duplicate=True),
    Output('wallet_graph','figure',allow_duplicate=True), # By default, objects should not used as output in multiple callbacks; have to allow duplicates manually
    Output('current_bitcoin_balance','value',allow_duplicate=True),
    Output('projection_bitcoin_balance','value',allow_duplicate=True),
    Output('current_bitcoin_usd_value','value',allow_duplicate=True),
    Input('wallet_address_submit_button','n_clicks'),
    State('wallet_addresses_text_input','value'),
    State('wallet_addresses','children'),
    prevent_initial_call=True
)
def update_portfolio_display(n_clicks,input_value,wallet_addresses):    
    if wallet_addresses is None:
        wallet_addresses = []
    if n_clicks > 0 and input_value:
        if validate_wallet(input_value): 
            input_value = '-'+str(input_value)
        else: 
            input_value = '-INVALID'+str(input_value)
        wallet_addresses.append(html.Div(input_value,style={'color': 'rgb(184, 134, 11)'}))
        input_value = ''
    fig = generate_graph(session_state.filtered_all_wallet_df)
    
    # Grab the current btc balance / usd value by querying these respective columns in the last row of the session_state's filtered_all_wallet_df
    curr_btc_balance = session_state.filtered_all_wallet_df['Amount'].iloc[-1]
    curr_usd_value = session_state.filtered_all_wallet_df['Value'].iloc[-1]
    curr_usd_value = f"${curr_usd_value:.2f}"

    return input_value,wallet_addresses,fig,curr_btc_balance,curr_btc_balance,curr_usd_value

# Callback to filter the wallet_graph similar to a traditional stock-chart; identify what timespan button was clicked by user and filter accordingly
@app.callback(
    Output('wallet_graph','figure',allow_duplicate=True),
    Input('1D_button','n_clicks'),
    Input('5D_button','n_clicks'),
    Input('1M_button','n_clicks'),
    Input('3M_button','n_clicks'),
    Input('6M_button','n_clicks'),
    Input('YTD_button','n_clicks'),
    Input('1Y_button','n_clicks'),
    Input('5Y_button','n_clicks'),
    Input('ALL_button','n_clicks'),
    prevent_initial_call=True)
def time_filter_graph(clicks_1d, clicks_5d, clicks_1m, clicks_3m, clicks_6m, clicks_ytd, clicks_1y, clicks_5y, clicks_all):
    # Check and store which input triggered the callback
    ctx = dash.callback_context
    button_id = None if not ctx.triggered else ctx.triggered[0]['prop_id'].split('.')[0]
        
    # Initialize the days_offset for filtering to 0; grab the earliest and latest dates in the session_state's filtered_all_wallet_df 
    days_offset = 0
    earliest_date = session_state.filtered_all_wallet_df['Date'].min()
    latest_date = session_state.filtered_all_wallet_df['Date'].max()

    # Determine days_offset by what button is clicked
    if button_id == '1D_button':
        days_offset = 1
    elif button_id == '5D_button':
        days_offset = 5
    elif button_id == '1M_button':
        days_offset = 30
    elif button_id == '3M_button':
        days_offset = 90
    elif button_id == '6M_button':
        days_offset = 180
    elif button_id == 'YTD_button':
        # Get the current year and highest date in dataframe (current date); this could be hardcoded, but I want it to be dynamic 
        current_year = datetime.now().year
    
        # Calculate the difference in days between january 1st of the same year and the latest date in the session_state's filtered_all_wallet_df dataframe
        start_of_year = datetime(current_year,1,1)
        days_offset = (latest_date - start_of_year.date()).days 
    elif button_id == '1Y_button':
        days_offset = 365
    elif button_id == '5Y_button':
        days_offset = 1825
    elif button_id == 'ALL_button':
        days_offset = (latest_date - earliest_date).days
    
    # Retrieve the day equal to the latest_date minus the days_offset number of days
    offset_days_back = (latest_date - pd.DateOffset(days=days_offset)).date()
    filtered_df = session_state.filtered_all_wallet_df[(session_state.filtered_all_wallet_df['Date'] >= offset_days_back) & (session_state.filtered_all_wallet_df['Date'] <= latest_date)]
    
    # Generate the graph including button_id as a parameter to distinguish from the initial call in the update_portflio_display function
    fig = generate_graph(filtered_df,button_id)
    return fig

# Callback to update the projection graph based on the user's selected projection year
### THIS IS A WORK IN PROGRESS AND WILL BE FINISHED NEXT WEEK; PLEASE SEE VIDEO AND ASSIGNMENT SUBMISSION FOR MORE DETAILS ###
@app.callback(
    Output('projection_graph_div','style'),
    Output('projection_graph','figure'),
    Input('regression_target_year','value'),
    State('wallet_addresses','children'),
    prevent_initial_call=True
)
def update_projection_display(input_year,wallet_addresses):
    # The below code is a placeholder for now, just returning an empty graph similar to the first graph when the page loads initially
    empty_df = pd.DataFrame([{'Date':datetime.today().date(),'Amount':0,'Value':0,'Price':0}])
    empty_fig = generate_graph(pd.DataFrame(),None,empty_df)
    if wallet_addresses==None:
        return {'display':'block'},empty_fig
    if input_year.isnumeric() and len(wallet_addresses)>0:
        # Call linear regression model here and generate graph based on generated dataframe; WORK IN PROGRESS
        return {'display':'block'},empty_fig
    return {'display':'block'},empty_fig

# Main function to run the Dash application / server
if __name__ == "__main__":
    #app.run_server(debug=True) # THIS LINE OR SOMETHING SIMILAR WILL BE USED IN OTHER IDE's; NOTE THAT THE APP IS FORMATTED FOR EXTERNAL WINDOWS
    # I HAVE ONLY USED JUPYTER_LABS FOR THIS ASSIGNMENT; I CANNOT SPEAK ON OTHER IDE's
    app.run_server(jupyter_mode='external',port=8993,debug=True)
    
# Misc. public Bitcoin wallet addresses to test (vary in loading time in the app; hit submit only once after pasting the wallet):
# 14bwkr3m8BWH8sgSXUiLVVS7CVEyHwz8sb, bc1q02mrh85muzdjk32sxu82022uke9qgjna6ydv05, 1MoooPejE6wvAcZxMo6KMBbwKeeTY2gmqN, 1FFcPEB7ZdUdmkhYmKnNwT6rTCY7jYNWnW, 17etp8Jgk2RqBZHLDWMHejMXwkfYJsk8FX

# One of Binance's main wallets: this one contains a huge amount of BTC (fun to test)
# 34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo


# In[ ]:




