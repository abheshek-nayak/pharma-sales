"""
Created on Sun Jun 10 21:48:32 2022

@author: ABHISHEK NAYAK
@copyright 2022 Abhishek Nayak

"""
# IMPORT ALL LIBRARIES

import pandas as pd
import datetime as dt
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymysql
import mysql.connector
from dash import Dash, html, dcc, Input, Output,dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from evalml.automl import AutoMLSearch
from evalml.model_understanding import graph_prediction_vs_actual_over_time

"""### CONNECT TO SQL """

app = Dash(external_stylesheets=[dbc.themes.MORPH])
app.title = 'XYZ Pharmaceuticals'

server = app.server

#SQL CLOUD DATABASE(Enter your own details)
#mydb = pymysql.connect(host="host ip", port=port_number, user="username", passwd="passowrd", database="database")

#SQL LOCALHOST
#=============================================================================
def DatabaseConnection(user,passwd,database):
    try:
	   mydb = mysql.connector.connect(host = 'localhost', user = 'your username',passwd = 'your passowrd',db = 'your database')
    except:
	   print("""The login credentials you entered are not valid for
 		   the database you indicated.  Please check your login details and try
 		   again.""")
    return mydb

mydb=DatabaseConnection('--username','--password','database')

#CREATE A CURSOR OBJECT
mycursor=mydb.cursor()
#=============================================================================

# READ THE DATABASE USING PANDAS
df=pd.read_sql("select datum,M01AB,M01AE,N02BA,N02BE,N05B,N05C,R03,R06 from salesdaily",mydb,index_col='datum',parse_dates=True )

"""### REMOVE OUTLIERS"""
#METHOD 1 -IQR
# =============================================================================
# cols = ['M01AB']
# 
# Q1 = df[cols].quantile(0.25)
# Q3 = df[cols].quantile(0.75)
# IQR = Q3 - Q1
# df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
# =============================================================================
# METHOD 2 : ZSCORE
from scipy.stats import zscore
zscore_ = df.loc[:,['M01AB']].apply(zscore).abs() < 3
df = df[zscore_.values]

"""### CHECK FOR NULL VALUES"""

df.isna().sum()
df.dropna(inplace=True)

df.reset_index(inplace = True)
df = df.rename(columns={'M01AB': 'y'})
X = pd.DataFrame(df["datum"])
y = df['y']


y_train, y_test= np.split(y, [int(.90 *len(y))])
X_train, X_test= np.split(X, [int(.90 *len(X))])

####DASH LAYOUT######
app.layout = html.Div([
    #html.Img(src='Your-compay-Logo'),
    html.H1('PHARMA SALES FORECAST',style={'textAlign': 'center',
                   
                   'fontSize': '55px',
                   'color': 'transparent',
                   'background-image': 'linear-gradient(45deg, #553c9a, #ee4b2b)', 
                   
                   'background-clip': 'text',
  '-webkit-background-clip': 'text',
    'font-weight':600
                   
  }),
    html.P('This dashboard shows the 7-30 day Sales Forcast',style={'textAlign': 'center',
                   
                   'fontSize': 20,
                   'color': 'black'}),
    html.Br(),
           
    dcc.RadioItems(id='pharma-dropdown',
                 options=['M01AB'],
                 value='M01AB',style={'width':'30%','verticalAlign':"middle"}),
    html.P('Choose a forecasting Period',style={'textAlign': 'left',
                   'fontFamily': 'Helvetica, sans-serif',
                   'fontSize': 15,
                   'color': 'black'}),
    
    dcc.Slider(id='forecast-dropdown',
                 min=7,max=30,step=2,value=7,tooltip={"placement": "bottom", "always_visible": True}),
    html.P('Choose a delay Period',style={'textAlign': 'left',
                   'fontFamily': 'Helvetica, sans-serif',
                   'fontSize': 15,
                   'color': 'black'}),
    dcc.Slider(id='delay-dropdown',
                 min=7,max=30,step=2,value=7,tooltip={"placement": "bottom", "always_visible": True}),
    
    html.Br(),

    html.H3('Forecasted Graph',style={'textAlign': 'center',
                   
                   'fontSize': 30,
                   'color': 'Navy'}),
    dcc.Graph(id='forcast-graph', figure={}),
      
       
        ],
    style={'padding': 100, 'border': 'solid','background-color': '#8EC5FC',
'background-image': 'linear-gradient(to top, #fbc2eb 0%, #a6c1ee 100%)'}
    )
   

@app.callback(    
    Output(component_id='forcast-graph', component_property='figure'),
   
    Input(component_id='forecast-dropdown', component_property='value'),
    Input(component_id='delay-dropdown', component_property='value'))

def fcast(fcast_horizon=7,m_delay=7):
    fcast_dict = {}
    fcast_dict["gap"]=0
    fcast_dict["max_delay"]=m_delay
    fcast_dict["forecast_horizon"]=fcast_horizon
    fcast_dict["time_index"]='datum'
    problem_config = fcast_dict
    automl = AutoMLSearch(X_train, y_train, problem_type="time series regression",
                          max_batches=1,
                          problem_configuration=problem_config,
                          allowed_model_families=["xgboost", "random_forest", "linear_model", "extra_trees"]
                          )
    automl.search()
    automl.best_pipeline
   
    baseline = automl.get_pipeline(0)
    baseline.fit(X_train, y_train)
 
    pipeline = automl.best_pipeline
    pipeline.fit(X_train, y_train)
    Pred = pipeline.predict(X_test.iloc[:pipeline.forecast_horizon], objective=None, X_train=X_train, y_train=y_train).rename('M01AB')
    Pred.index = list(range(len(Pred)))
    Pred.rename(index={0: 'Tommorow'},inplace=True)
    
    fig = px.line(Pred,markers=True,labels={'index': 'days'})
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    






