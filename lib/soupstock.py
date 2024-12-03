#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###<HEADER_>==================================================================#
"""
Created on Mon Oct 14 16:42:20 2024

@author: dac
"""
###############################################################################
## Webscraping
# !pip install pandas
# !pip install requests
# !pip install bs4
# !pip install html5lib
# !pip install lxml
# !pip install plotly

import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings ; ## Ignore all warnings
warnings.filterwarnings("ignore", category=FutureWarning) ;

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_graph(stock_data, revenue_data, stock):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing = .3)
    stock_data_specific = stock_data[stock_data.Date <= '2021--06-14']
    revenue_data_specific = revenue_data[revenue_data.Date <= '2021-04-30']
    fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data_specific.Date), y=stock_data_specific.Close.astype("float"), name="Share Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data_specific.Date), y=revenue_data_specific.Revenue.astype("float"), name="Revenue"), row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
    fig.update_layout(showlegend=False,
    height=900,
    title=stock,
    xaxis_rangeslider_visible=True)
    fig.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def get_stock( stock_name ):
    stock_all = yf.Ticker( stock_name );
    stock_history = stock_all.history( period="max");    
    stock_history.reset_index(inplace=True);
    return stock_history;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_html( url ):
    r = requests.get( url );
    html_data = ( r ).text ;
    return html_data;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_soup( html_data ):
    soup = BeautifulSoup( html_data, 'html.parser') ; 
    return soup ;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def table_revenue( soup):
    tables = soup.find('tbody')
    Date = list(); Revenue = list();
    for row in tables.find_all('tr'):
        col = row.find_all('td');
        Date.append( col[0].text) ;
        Revenue.append( col[1].text ); 
    df = pd.DataFrame( 
        {
        "Date":     Date, 
        "Revenue" : Revenue,
        } 
    );
    df["Revenue"] = df['Revenue'].str.replace(r',|\$',"", regex=True);
    print( df.head() );
    print( df.tail() );
    return df;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if '__main__'==__name__:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    url = {
        'GME': 
          'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'+\
            'IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/stock.html',
        'TSLA':
          'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'+\
            'IBMDeveloperSkillsNetwork-PY0220EN-SkillsNetwork/labs/project/revenue.htm',
        }
    
    for stock_name in ['TSLA', 'GME']:
        stock_data  = get_stock( stock_name )
        soup        = get_soup ( get_html( url[stock_name] ));
        revenue_data= table_revenue( soup );
        make_graph( stock_data, revenue_data, stock_name );
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

