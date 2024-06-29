import numpy as np
import pandas as pd
import os
import plotly.express as px

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from plotly.subplots import make_subplots
import plotly.graph_objs as go


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from datetime import datetime

FREQ = [5,10,15]

import warnings
warnings.filterwarnings('ignore')


_dict_weekdays = {0:'Monday', 1:'Tuesday', 2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday', 6:'Sunday'}


class LoadTransformCryptoData():
    
    def __init__(self,minutes=FREQ):
        self.minutes = minutes
        self.load_crypto_data()
        self.resample()
                
    def load_crypto_data(self):
        _dict = {}
        for file in os.listdir('data'):
            if '.csv' in file:# and ('ADA' in file or 'ALGO' in file):
                _df = pd.read_csv(f'data/{file}',delimiter=',')
                _df['timestamp']= _df['timestamp'].apply(lambda x:  datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                _dict[file.split('_')[0]] = _df.set_index('timestamp')

        self.raw_data=_dict.copy()
        
    # check for missing data points? minutes?
    def resample(self):

        _dict = {}
        _dict[1] = {}
        for minutes in self.minutes:
            _dict[minutes]={}
            
            for k, df in self.raw_data.items():
                _df = df.copy()
                #add main variables
                _df['close%'] = _df['close']/_df['open']-1
                _df['volume%'] = _df['volume']/_df['volume'].shift(1)-1
                _dict[1][k] = _df.copy()

                df_resampled = df.resample(f'{minutes}min',label='right').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
#                     'close': 'last',
                    'volume': 'sum'
                })
                df_resampled = df_resampled.merge(df[['close']],left_index=True,right_index=True)[['open','high','low','close','volume']]

                #add main targets:
                df_resampled['close%'] = df_resampled['close']/df_resampled['open']-1
                df_resampled['volume%'] = df_resampled['volume']/df_resampled['volume'].shift(1)-1

                _dict[minutes][k] = df_resampled.copy()

        print(f"Resampled Data ({self.minutes} minutes):")

        self.resampled_data = _dict.copy()

        
reload = False

if reload is True:
    crypto_data = LoadTransformCryptoData()

    crypto_raw = crypto_data.raw_data.copy()
    crypto_resampled = crypto_data.resampled_data.copy()
    %store crypto_raw
    %store crypto_resampled
    
else:
    %store -r crypto_raw
    %store -r crypto_resampled

freq = 15
crypto_resampled[freq]['ADA'].head()


class FeatureEngineering():
    """
    preselected resampled frequency
    
    import talib
    https://ta-lib.org/functions/
    
    
    """
    
    def __init__(self,_dict,nlags = range(1,20),windows= [3,5,10,20]):
        self.windows = windows
        self.nlags = nlags
        self._dict = _dict.copy()
        
        
    def run(self):
        _dict = self._dict.copy()
        
        for crypto,df in _dict.items():
            df_target = self.add_target(df)
            df_time = self.time_features(df_target)
            df_lags = self.get_lags(df_time)
            df_rol = self.rolling_window_same_period(df_lags)
            df_add = self.additional(df_rol)
            
            _dict[crypto] = df_add.copy()
            
        self.crypto_added_features = _dict.copy()
        
        self.df_modelling()
        
    # add main currencies changes..
    def add_target(self,df):
        df['up_dummy'] = df['close%'].apply(lambda x: 1 if x>0 else 0)  #dummy increaase = 1, decrease=1 or classification?
        df['up_class'] = df['close%'].apply(lambda x: 2 if x<=-0.02 else 1 if x>=0.02 else 0)  #dummy increaase = 1, decrease=1 or classification?
        # clean[['close%']][::-1].rolling(3).mean()
        
        return df
        
    
    def time_features(self,df):
        df['year'] = df.index.year
        df['year'] = df['year'].apply(lambda x: 1 if x == 2024 else 0)
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.weekday
        df['weekday'] = df['day_of_week'].apply(lambda x: 1 if x<5 else 0)
        df['day_of_week'] = df['day_of_week'].apply(lambda x: _dict_weekdays[x])
        df['hour'] = df.index.hour
        df = pd.get_dummies(df, columns=['day_of_week'],dtype=float)
        return df
    
    
    #e-	Lags: #AUC & PAUC to be done 
    def get_lags(self,df,columns = ['close%']):
        for column in columns:
            for n in self.nlags:
                df[f'{column}_{n}lags']= df[column].shift(n)

        return df

    #-	Rolling window statistics - same frequency based
    def rolling_window_same_period(self,df):
        for window in self.windows:
            df[f'rol_mean_close_{window}'] = df['close%'].rolling(window).mean()
            df[f'rol_mean_volume_{window}'] = df['volume%'].rolling(window).mean()
            df[f'rol_std_close_{window}'] = df['close%'].rolling(window).std()
            df[f'rol_std_volume_{window}'] = df['volume%'].rolling(window).std()
            
        return df

    #-	Rolling window statistics - within windows based?

    # -	Harmonic decomposition, Fourier
    #additional features

    def additional(self,df):
        df['HLC']=df['high']/df['low'] - 1
        df['HL']=(df['high']-df['low'])/df['close']
        
        return df
    
    def df_modelling(self):
        _dict_clean = {}
        for crypto, df in self.crypto_added_features.items():
            _dict_clean[crypto] = df.drop(columns = ['open','high','low','close','volume']).dropna()
        
        self.crypto_modelling = _dict_clean.copy()
        
        
freq = 15
working_dict = crypto_resampled[freq].copy()

features = FeatureEngineering(_dict=working_dict)
features.run()
features.crypto_modelling["ADA"].head()


class CustomTimeSeriesSplit:
    
    
    def __init__(self,X, train_size, test_size):
        self.train_size = train_size
        self.test_size = test_size
        self.X = X
        
    def run(self):
        self.split()
        self.get_n_splits()
        

    def split(self):
        n_samples = len(self.X)
        indices = np.arange(n_samples)
        splits = []
        
        start_train = 0
        # to do: last unused set (remainig - used for validation at the end for parameter tuning?)
        while start_train + self.train_size + self.test_size <= n_samples:
            end_train = start_train + self.train_size
            start_test = end_train
            end_test = start_test + self.test_size
            
            train_indices = indices[start_train:end_train]
            test_indices = indices[start_test:end_test]
            
            splits.append((df.index[train_indices], df.index[test_indices]))
            
            start_train = end_test  # move the window to the next non-overlapping position

        self.splits = splits

    def get_n_splits(self, y=None, groups=None):
        n_samples = len(self.X)
        self.n_splits = (n_samples - self.train_size) // (self.train_size + self.test_size)

        

# next x periods

class CVresults():
    """
    per crypto
    
    """
    
    def __init__(self,model,crypto,splits):
        self.model=model
        self.crypto=crypto
        self.splits=splits
        self.df = features.crypto_modelling[self.crypto]
        
    def run(self):
        self.calculate_test_results()
        self.transform_results()
        self.create_actual_forecast()
    
    
    def calculate_test_results(self):
        
        test_results = []
        
        for cv_set in range(0,len(self.splits)):

            x_train=np.array(self.df[EXOG].reindex(self.splits[cv_set][0]))
            x_test = np.array(self.df[EXOG].reindex(self.splits[cv_set][1]))


            #linear
            if self.model == 'linear':
                endog = 'close%'
                y_train=np.array(self.df[endog].reindex(self.splits[cv_set][0]).values)
                y_test = self.df.reindex(self.splits[cv_set][1])[endog].values[0]
                
                reg = LinearRegression().fit(x_train, y_train)
                R = reg.score(x_train, y_train)
                test_results.append([self.splits[cv_set][1][0] #time
                                     , reg.predict(x_test)[0] #pred
                                     , y_test #actual
#                                      ,R
                                    ]) #index, predicted, actual, R
                
            if self.model == 'random_forest':
                endog = 'close%'
                y_train=np.array(self.df[endog].reindex(self.splits[cv_set][0]).values)
                y_test=self.df.reindex(self.splits[cv_set][1])[endog].values[0]

                regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
                regressor.fit(x_train, y_train)

                # Making predictions on the same data or new data
                predictions_test = regressor.predict(x_test)
                
                test_results.append([self.splits[cv_set][1][0]
                     , predictions_test[0]
                     , y_test
#                                      ,R
                    ]) #index, predicted, actual, R


        assert len(test_results)== len(self.splits)
                
        self.test_results = test_results
        
        
    def transform_results(self):
        freq=15
        df = pd.DataFrame(self.test_results,columns = ['timestamp','predicted%','actual%']).set_index('timestamp')
        #join actual from resampled, open
        price_open = crypto_resampled[freq][self.crypto][['open','close']]
        
        converted = df.merge(price_open, how='left',left_index=True,right_index=True).rename(columns = {'close':'actual_price'})
        converted['predicted_price'] = converted['open']*(1+converted['predicted%'])
        
        self.converted = converted.copy()
        
    def create_actual_forecast(self):
        df = self.converted.copy()
        df['actual%'] = df['actual%']*100
        df['predicted%'] = df['predicted%']*100
        df['d'] = df.apply(lambda x: 1 if x['actual%']*x['predicted%'] > 0 else 0,axis=1)
        D = np.sum(df['d'])*(100/(len(df)-1))
        MSE =  1/(len(df) - 1)* np.sum((df['actual%'] - df['predicted%'])**2)
        MAPE = 100/len(df) * np.sum((df['actual_price'] - df['predicted_price'])/df['actual_price'])
        
        #percentage
        fig_perc = px.line(df[['actual%','predicted%']])
        fig_perc.update_layout(title_text=F'Crypto:{self.crypto}, actual vs forecast (%)for {len(df)} cv sets<br>D:{np.round(D,1)}%, MAPE:{np.round(MAPE,2)}'
                                ,width = 1500
                                ,height = 500
                                , template = 'presentation'
                                ,showlegend = True
                                ,font = {'size':15}
                                             , xaxis = {"title": '','showgrid':False}
                                             , yaxis = {"title": ''}
                                ,legend = {'orientation':"h",'yanchor':"top",'title':''}
                         ).update_yaxes(title_text = '', secondary_y = False).update_yaxes(title_text = "",showgrid = False ,secondary_y=True)
        
        #price
        fig_price = px.line(df[['actual_price','predicted_price']])
        fig_price.update_layout(title_text=F'Crypto:{self.crypto}, actual vs forecast price for {len(df)} cv sets<br>D:{np.round(D,1)}%, MAPE:{np.round(MAPE,2)}'
                                ,width = 1500
                                ,height = 500
                                , template = 'presentation'
                                ,showlegend = True
                                ,font = {'size':15}
                                             , xaxis = {"title": '','showgrid':False}
                                             , yaxis = {"title": ''}
                                ,legend = {'orientation':"h",'yanchor':"top",'title':''}
                         ).update_yaxes(title_text = '', secondary_y = False).update_yaxes(title_text = "",showgrid = False ,secondary_y=True)

        self.fig_perc = fig_perc
        self.fig_price = fig_price
        self.stats = {'D%':D,'MSE%':MSE,'MAPE':MAPE}


EXOG = [i for i in features.crypto_modelling[crypto].columns if 'rol' in i] #not in ['close%','up_dummy','up_class','year','month']]


# cretion of CV sets
crypto = 'ADA'
df=features.crypto_modelling['ADA'].copy()

cst = CustomTimeSeriesSplit(X=df.index, train_size=500, test_size=1)
cst.run()

splits = {}
for i, (train_index, test_index) in enumerate(cst.splits):
    splits[i] = [train_index,test_index]
    
    
# creation of test results
cv_results = CVresults(model='random_forest',crypto=crypto, splits=splits)
cv_results.run()
cv_results.fig_perc