#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings;warnings.simplefilter(action='ignore', category=FutureWarning);# Suppress FutureWarning
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import myfavs as my;
import danpy as dan;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# basics
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import matplotlib as mpl;
import seaborn as sns;
from scipy import stats;
# %matplotlib inline # uncomment only in jupyter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# sci-kit modeling
from sklearn.metrics import mean_squared_error, r2_score;
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,GridSearchCV;
from sklearn.preprocessing import PolynomialFeatures, StandardScaler;
from sklearn.pipeline import Pipeline;
from sklearn.linear_model import LinearRegression, Ridge;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import requests; ## for json webscrapping requests
import folium;   ## maps
import json;

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dashboards
import plotly.express as px ;
import dash ;
from dash import html ;
from dash import dcc ;
from dash.dependencies import Input, Output ;
#import dash_html_components as html ; ## deprecated
#import dash_core_components as dcc ;  ## deprecated
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings ; ## Ignore all warnings
warnings.filterwarnings("ignore", category=FutureWarning) ;