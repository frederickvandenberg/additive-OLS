##### IMPORTATION REQUIRMENTS (PYTHON)

# BASE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# MODELLING
from sklearn.linear_model import LinearRegression
from scipy import stats
import scipy.stats
from scipy.signal import savgol_filter
from scipy.stats import zscore

# CONFIDENCE INTERVALS
from numpy import sum as arraysum
from numpy import sqrt

# VALIDATION
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.stattools import grangercausalitytests

# CUSTOM QUERY
from utils.config import config_dict
import utils.db_toolbox as tb

# DATETIME
from datetime import datetime, timedelta
import datetime
import calendar

# PLOTLY
import plotly.figure_factory as ff
import plotly.offline as pyo
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table

# MONTH ADDITIVE FUNCTION
def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day)

##### SQL QUERY (FREQUENCY, PRICE)

# PARAMETERS
element = 'gold'
date_from = '2000-01-01'
date_to = '2020-09-30'

# ESTABLISH CONNECTION
con = tb.db_con(config_dict)

# FREQUENCY QUERY
df = pd.DataFrame(con.read_query(f"""select pub_date, heading, sub_heading, sentiment
                                    from articles
                                    where unique_text like '%{element}%' AND pub_date BETWEEN '{date_from}' AND '{date_to}'
                                    order by pub_date desc;"""),
                                    columns=['pub_date','heading','sub_heading','sentiment'])

# DATAFRAME MODIFICATION
df['combined'] = df['heading'] + '. ' + df['sub_heading']
del df['heading']
del df['sub_heading']

# PRICE QUERY
pf = pd.DataFrame(con.read_query(f"""select spot_date, am_price
                                    from metal_price
                                    where (commodity like '%{element}%' AND spot_date BETWEEN '{date_from}' AND '{date_to}')
                                    order by spot_date desc;"""),
                                    columns=['spot_date','am_price'])
 
# DATETIME FORMATTING
df['pub_date'] = pd.to_datetime(df['pub_date'])
pf['spot_date'] = pd.to_datetime(pf['spot_date'])
df.index = df['pub_date']
df.index = pd.to_datetime(df.index)
pf.index = pf['spot_date']
pf.index = pd.to_datetime(pf.index)

# RESAMPLE DATAFRAMES TO ONE
time_period = 'M'
dff = df['combined'].resample(time_period).count().rename('f')
dfs = df['sentiment'].resample(time_period).mean().rename('s')
pfp = pf['am_price'].resample(time_period).mean().rename('p')
df = pd.concat([dff,dfs,pfp], axis=1)
df['date']=df.index
df['year']=df.index.year
df['month']=df.index.month
df['week']=df.index.week
df.dropna(inplace=True)

###### ADDITIVE OLS LINEAR MODEL

# PRE-SET PARAMETERS
prediction = []
date =[]
lower = []
upper = []
r = []
R = []
coef = []
intercept = [[]]
cork = 144 # 144 ILOC EQUIVILENT TO 2012-01-01, CORK MEANS START OF MODEL DATA
end = 151

# LOOPING ADDITIVE ITERATION
for i in range(end,len(df)):
    shift = 7 # MONTHS

    # MODEL TRAIN-TEST SPLIT
    X = df['f'].shift(shift).iloc[cork:end].to_numpy().reshape(-1, 1)
    y = df['p'].iloc[cork:end].to_numpy()

    # MODEL DEFINITION
    reg = LinearRegression().fit(X, y)

    # PRINT RESULTS
    print('--Parameters--')
    print('Date: ', df['date'].iloc[cork],'-TO-', df['date'].iloc[end])
    print(f'Lag: {shift} {time_period}')
    print('r: ', np.sqrt(reg.score(X, y)))
    print('R^2: ', reg.score(X, y))
    print('Coefficient: ', (reg.coef_)[0])
    print('Intercept: ', reg.intercept_)

    # PLOT RESULTS
    plt.scatter(X, y,color='b')
    plt.plot(X, reg.predict(X),color='r')
    plt.show();
    
    # CONFIDENCE INTERVALS
    yhat = float(reg.predict(np.asarray([[df['f'][end]]], dtype=np.float32).reshape(-1, 1)))
    sum_errs = arraysum((y - yhat)**2)
    stdev = sqrt(1/(len(y)-2) * sum_errs)
    yhat_out = yhat
    interval = 1 * stdev
    lower_value, upper_value = yhat_out - interval, yhat_out + interval
    
    # PREDICTION & CONFIDENCE LISTS APPEND
    prediction.append(float(reg.predict(np.asarray([[df['f'][end]]], dtype=np.float32).reshape(-1, 1))))
    date.append(add_months(df.index[end],shift))
    lower.append(lower_value)
    upper.append(upper_value)
    
    # EVAL LIST APPEND
    r.append(np.sqrt(reg.score(X, y)))
    R.append(reg.score(X, y))
    coef.append((reg.coef_)[0])
    intercept.append(reg.intercept_)
    
    # INCREMENT LOOP
    end += 1


# PREDICTION & CONFIDENCE INTERVAL DATAFRAME CREATION
results = pd.DataFrame(list(zip(prediction,upper,lower,date)),columns=['prediction','upper','lower','date'])
results['date'] =pd.to_datetime(results['date'])
results.index=results['date']
results['year']=results.index.year
results['month']=results.index.month
results['series'] = range(1, len(results) + 1)

# EVA: DATAFRAME CREATION
evaluation = pd.DataFrame(list(zip(r,R,coef,intercept,date)),columns=['r','R','coef','intercept','date'])
evaluation['date'] = pd.to_datetime(evaluation['date'])
evaluation.index = evaluation['date']
evaluation['year']=evaluation.index.year
evaluation['month']=evaluation.index.month
evaluation['series'] = range(1, len(evaluation) + 1)

#### POLYNOMIAL ITERATION
time1 = '2013-01-01'
time2 = '2015-06-01'
count = 1
time1 = datetime.datetime.strptime(time1, '%Y-%m-%d')
time2 = datetime.datetime.strptime(time2, '%Y-%m-%d')

prediction_date = []
prediction_year = []
prediction_month = []
raw_prediction = []
raw_upper_ci = []
raw_lower_ci = []
mean_prediction = []
mean_upper_ci = []
mean_lower_ci = []
publish_date = []
series=[]

while count < 79:
    
    # SLICING
    temp_results = results.loc[time1:time2]
    temp_results['series'] = range(1, len(temp_results) + 1)
    
    # POLYNOMIAL FIT
    for i in range(2,11):
        x = temp_results['series']
        y = temp_results['prediction']
        xp = np.linspace(1, len(temp_results), len(temp_results))
        p = np.poly1d(np.polyfit(x, y, i))
        temp_results[f'fit_prediction_{i}'] = p(xp) 
        
    # UPPER CI
    for i in range(2,11):  
        x = temp_results['series']
        y = temp_results['upper']
        xp = np.linspace(1, len(temp_results), len(temp_results))
        p = np.poly1d(np.polyfit(x, y, i))
        temp_results[f'fit_upper_{i}'] = p(xp)
        
    # LOWER CI
    for i in range(2,11):
        x = temp_results['series']
        y = temp_results['lower']
        xp = np.linspace(1, len(temp_results), len(temp_results))
        p = np.poly1d(np.polyfit(x, y, i))
        temp_results[f'fit_lower_{i}'] = p(xp)

    # MEAN POLYNOMIAL FIT
    temp_results['fit_mean'] = temp_results[['fit_prediction_2','fit_prediction_3','fit_prediction_4',
                               'fit_prediction_5','fit_prediction_6','fit_prediction_7','fit_prediction_8',
                              'fit_prediction_9','fit_prediction_10']].mean(axis=1)

    temp_results['upper_mean'] = temp_results[['fit_upper_2','fit_upper_3','fit_upper_4',
                               'fit_upper_5','fit_upper_6','fit_upper_7','fit_upper_8',
                              'fit_upper_9','fit_upper_10']].mean(axis=1)

    temp_results['lower_mean'] = temp_results[['fit_lower_2','fit_lower_3','fit_lower_4',
                               'fit_lower_5','fit_lower_6','fit_lower_7','fit_lower_8',
                              'fit_lower_9','fit_lower_10']].mean(axis=1)
    
    # DROP 7TH PREDICTION
    temp_results = temp_results[22:28]
    temp_results['series'] = range(1, len(temp_results) + 1)
    
    for i in range(0,len(temp_results)):
        temp_results['publish_date'] = add_months(time2,-6)
    
    # PREDICTION & CONFIDENCE LISTS EXTEND
    prediction_date.extend(temp_results['date'].tolist())
    prediction_year.extend(temp_results['year'].values)
    prediction_month.extend(temp_results['month'].values)
    raw_prediction.extend(temp_results['prediction'].values)
    raw_upper_ci.extend(temp_results['upper'].values)
    raw_lower_ci.extend(temp_results['lower'].values)
    mean_prediction.extend(temp_results['fit_mean'].values)
    mean_upper_ci.extend(temp_results['upper_mean'].values)
    mean_lower_ci.extend(temp_results['lower_mean'].values)
    publish_date.extend(temp_results['publish_date'].values)
    series.extend(temp_results['series'].values)
    
    # ADD ONE MONTH
    time1 = add_months(time1,1)
    time2 = add_months(time2,1)

    # OUTPUT CHECK
    print(time1)
    print(time2)
    print(count)
    count += 1

# PREDICTION & CONFIDENCE INTERVAL DATAFRAME CREATION
poly_prediction = pd.DataFrame(list(zip(prediction_date,
                                        prediction_year,
                                        prediction_month,
                                        raw_prediction,
                                       raw_upper_ci,
                                       raw_lower_ci,
                                       mean_prediction,
                                       mean_upper_ci,
                                       mean_lower_ci,
                                       publish_date,
                                       series)),
                               columns=['prediction_date',
                                        'prediction_year',
                                        'prediction_month',
                                        'raw_prediction',
                                       'raw_upper_ci',
                                       'raw_lower_ci',
                                       'mean_prediction',
                                       'mean_upper_ci',
                                       'mean_lower_ci',
                                       'publish_date',
                                       'series'])

 ##### PLOTLY VISUALS
fig = go.Figure()
#'2020-06-01'


fig.add_trace(go.Scatter(x=poly_prediction['prediction_date'],
                         y=poly_prediction['mean_prediction'], 
                        marker=dict(
                            size=10,
                            color=poly_prediction['series'],
                            colorbar=dict(
                                title="Colorbar"
                            ),
                            colorscale="reds"
                        ),
                    mode='markers',
                    line=dict(color='grey', width=1),
                    name='LCI'))

fig.add_trace(go.Scatter(x=df.loc['2015-01-01':].index, y=df['p'].loc['2015-01-01':],
                    mode='lines',
                    line=dict(color='#343A40', width=2),
                    name='Gold Price'))


fig.update_layout(
    #title="Plot Title",
    #xaxis_title="x Axis Title",
    yaxis_title="Gold Price ($/oz)",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
    width=1000,
    height=600,
    font=dict(
        family="Arial",
        size=12,
        color="#7f7f7f"
    ))

fig.update_xaxes(showline=True, linewidth=2, linecolor="#7f7f7f",gridcolor='rgba(250,250,250,0.9)')
fig.update_yaxes(showline=True, linewidth=2, linecolor="#7f7f7f",gridcolor='rgba(250,250,250,0.9)')

fig.show()
fig.write_image("report_gold.svg") 


##### TREND EMPERICAL PROBABILITY

# all time
# Time 1
time1 = '2015-01-01'

# Time 2
time2 = '2020-08-31'

##### 3 YEAR
# Time 3
time3 = '2016-11-01'

# Time 4
time4 = '2020-05-30'

##### 1 YEARS
# Time 5
time5 = '2018-11-01'

# Time 6
time6 = '2020-05-30'

time_period = 'M'
temp_df = poly_prediction
temp_df['prediction_date'] = pd.to_datetime(temp_df['prediction_date'])
temp_df.index=temp_df['prediction_date']
mean_mean_prediction = temp_df['mean_prediction'].loc[time1:time2].resample(time_period).mean().rename('mean_mean_prediction')
mean_upper_prediction = temp_df['mean_upper_ci'].loc[time1:time2].resample(time_period).mean().rename('mean_upper_prediction')
mean_lower_prediction = temp_df['mean_lower_ci'].loc[time1:time2].resample(time_period).mean().rename('mean_lower_prediction')
ppp = df['p'].loc[time1:time2].resample(time_period).mean().rename('price')

model_results = pd.DataFrame(list(zip(mean_mean_prediction,
                                        mean_upper_prediction,
                                        mean_lower_prediction,
                                         ppp)),
                               columns=['fit_prediction',
                                        'fit_upper',
                                        'fit_lower',
                                        'price'
                                       ], index=mean_mean_prediction.index)
                                  
model_results.to_csv('gold_prediction_results.csv')

model_results_pct = model_results.pct_change().dropna()
model_results_pct.plot(figsize=(12,8))

model_results_pct['trend'] = 'a'

for i in model_results_pct.index:
    if model_results_pct['price'][i] > 0 and model_results_pct['fit_prediction'][i] > 0:
        model_results_pct['trend'][i] = 1
    elif model_results_pct['price'][i] < 0 and model_results_pct['fit_prediction'][i] < 0:
        model_results_pct['trend'][i] = 1
    else:
        model_results_pct['trend'][i] = 0

# TREND CHECK (%)

# 2015 to Present
print('>>>2015 to present')
print(model_results_pct['trend'].loc['2015-02-01':'2020-06-30'].sum()/len(model_results_pct['trend'].loc['2015-02-01':'2020-06-30'])*100)
# 2016 to Present
print('>>>2016 to present')
print(model_results_pct['trend'].loc['2016-01-01':'2020-06-30'].sum()/len(model_results_pct['trend'].loc['2016-01-01':'2020-06-30'])*100)
# 2017 to Present
print('>>>2017 to present')
print(model_results_pct['trend'].loc['2017-01-01':'2020-06-30'].sum()/len(model_results_pct['trend'].loc['2017-01-01':'2020-06-30'])*100)
# 2018 to Present
print('>>>2018 to present')
print(model_results_pct['trend'].loc['2018-01-01':'2020-06-30'].sum()/len(model_results_pct['trend'].loc['2018-01-01':'2020-06-30'])*100)
# 2019 to Present
print('>>>2019 to present')
print(model_results_pct['trend'].loc['2019-01-01':'2020-06-30'].sum()/len(model_results_pct['trend'].loc['2019-01-01':'2020-06-30'])*100)

##### Yearly trend check (%)
# TREND CHECK (%)

# 2015 to Present
print('>>>2015 to 2016')
print(model_results_pct['trend'].loc['2015-01-01':'2016-01-01'].sum()/len(model_results_pct['trend'].loc['2015-02-01':'2016-01-01'])*100)
# 2016 to Present
print('>>>2016 to 2017')
print(model_results_pct['trend'].loc['2016-01-01':'2017-01-01'].sum()/len(model_results_pct['trend'].loc['2016-01-01':'2017-01-01'])*100)
# 2017 to Present
print('>>>2017 to 2018')
print(model_results_pct['trend'].loc['2017-01-01':'2018-01-01'].sum()/len(model_results_pct['trend'].loc['2017-01-01':'2018-01-01'])*100)
# 2018 to Present
print('>>>2018 to 2019')
print(model_results_pct['trend'].loc['2018-01-01':'2019-01-01'].sum()/len(model_results_pct['trend'].loc['2018-01-01':'2019-01-01'])*100)
# 2019 to Present
print('>>>2019 to 2020')
print(model_results_pct['trend'].loc['2019-01-01':'2020-01-01'].sum()/len(model_results_pct['trend'].loc['2019-01-01':'2020-01-01'])*100)
# 2020 to 2021
print('>>>2020 to 2021')
print(model_results_pct['trend'].loc['2020-01-01':'2020-06-30'].sum()/len(model_results_pct['trend'].loc['2020-01-01':'2020-06-30'])*100)

##### TIMESERIES ERROR METRICS

##### ROOT MEAN SQUARED ERROR
# ALL
print('>5 YEARS')
RMSE = rmse(df['p'].loc[time1:time2], mean_mean_prediction.loc[time1:time2])
print(RMSE)
print('>3 YEARS')
RMSE = rmse(df['p'].loc[time3:time4], mean_mean_prediction.loc[time3:time4])
print(RMSE)
print('>1 YEAR')
RMSE = rmse(df['p'].loc[time5:time6], mean_mean_prediction.loc[time5:time6])
print(RMSE)


##### mean_squared_error
from sklearn.metrics import mean_squared_error

# ALL
print('>5 YEARS')
MSE = mean_squared_error(df['p'].loc[time1:time2], mean_mean_prediction.loc[time1:time2])
print(MSE)
print('>3 YEARS')
MSE = mean_squared_error(df['p'].loc[time3:time4], mean_mean_prediction.loc[time3:time4])
print(MSE)
print('>1 YEAR')
MSE = mean_squared_error(df['p'].loc[time5:time6], mean_mean_prediction.loc[time5:time6])
print(MSE)


##### median_absolute_error
from sklearn.metrics import median_absolute_error

# ALL
print('>5 YEARS')
MAE = median_absolute_error(df['p'].loc[time1:time2], mean_mean_prediction.loc[time1:time2])
print(MAE)
print('>3 YEARS')
MAE = median_absolute_error(df['p'].loc[time3:time4], mean_mean_prediction.loc[time3:time4])
print(MAE)
print('>1 YEAR')
MAE = median_absolute_error(df['p'].loc[time5:time6], mean_mean_prediction.loc[time5:time6])
print(MAE)

##### PLOTLY VISUALS
fig = go.Figure()
#'2020-06-01'

fig.add_trace(go.Scatter(x=mean_upper_prediction.loc['2017-11-01':].index, y=mean_upper_prediction.loc['2017-11-01':].loc[:],
                    mode='lines',
                    line=dict(color='grey', width=1),
                    name='LCI'))

fig.add_trace(go.Scatter(x=mean_lower_prediction.loc['2017-11-01':].index, y=mean_lower_prediction.loc['2017-11-01':].loc[:],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(240,240,240,0.25)',
                    line=dict(color='grey', width=1),
                    name='UCI'))

fig.add_trace(go.Scatter(x=mean_mean_prediction.loc['2017-11-01':].index,
                         y=mean_mean_prediction.loc['2017-11-01':], 
                    mode='lines',
                    line=dict(color='gold', width=2),
                    name='Prediction'))

fig.add_trace(go.Scatter(x=df.loc['2017-11-01':].index, y=df['p'].loc['2017-11-01':],
                    mode='lines',
                    line=dict(color='#343A40', width=2),
                    name='Actual'))


fig.update_layout(
    #title="Plot Title",
    #xaxis_title="x Axis Title",
    yaxis_title="Gold Price ($/oz)",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
    width=900,
    height=600,
    font=dict(
        family="Arial",
        size=12,
        color="#7f7f7f"
    ))

fig.update_xaxes(showline=True, linewidth=2, linecolor="#7f7f7f",gridcolor='rgba(250,250,250,0.9)')
fig.update_yaxes(showline=True, linewidth=2, linecolor="#7f7f7f",gridcolor='rgba(250,250,250,0.9)')

fig.show()
fig.write_image("gold_prediction_linegraph.svg")

# Yearly barchart
time_period = 'Y'
yfp = model_results['fit_prediction'].loc['2015-01-01':'2020-06-30'].resample(time_period).mean().rename('y_fit_prediction')
ufp = model_results['fit_upper'].loc['2015-01-01':'2020-06-30'].resample(time_period).mean().rename('y_upper_prediction')
lfp = model_results['fit_lower'].loc['2015-01-01':'2020-06-30'].resample(time_period).mean().rename('y_lower_prediction')
ppp = df['p'].loc['2015-01-01':'2020-06-30'].resample(time_period).mean().rename('y_price')

y_model_results = pd.DataFrame(list(zip(yfp,
                                        ufp,
                                        lfp,
                                       ppp)),
                               columns=['yfp',
                                        'ufp',
                                        'lfp',
                                       'price'], index=ppp.index.year)


trace1 = go.Bar(x=y_model_results.index,
                y=y_model_results['yfp'],
                name='Prediction',
               marker={'color':'gold'})

trace2 = go.Bar(x=y_model_results.index,
                y=y_model_results['price'],
                name='Actual',
               marker={'color':'#343A40'})

data = [trace1,trace2]
layout = go.Layout(title='Metrics')
fig = go.Figure(data=data,layout=layout)
fig.update_layout(
    #title="Plot Title",
    #xaxis_title="x Axis Title",
    yaxis_title="Gold Price ($/oz)",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
    width=900,
    height=600,
    font=dict(
        family="Arial",
        size=12,
        color="#7f7f7f"
    ))
fig.show()
fig.write_image("gold_prediction_bargraph.svg")