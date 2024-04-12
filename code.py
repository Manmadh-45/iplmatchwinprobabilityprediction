import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

print(matches.head())

print(deliveries.head())

sum_score = deliveries.groupby(['match_id','inning']).sum()['total_runs'].reset_index()

sum_score = sum_score[sum_score['inning'] == 1]

matches_data = matches.merge(sum_score[['match_id','total_runs']],left_on='id',right_on='match_id')

print(matches_data)

print(matches_data['team1'].unique())

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Gujarat Titans',
    'Lucknow Super Giants',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

matches_data['team1'] = matches_data['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
matches_data['team2'] = matches_data['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

matches_data['team1'] = matches_data['team1'].str.replace('Kings XI Punjab','Punjab Kings')
matches_data['team2'] = matches_data['team2'].str.replace('Kings XI Punjab','Punjab Kings')

matches_data['team1'] = matches_data['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
matches_data['team2'] = matches_data['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

matches_data = matches_data[matches_data['team1'].isin(teams)]
matches_data = matches_data[matches_data['team2'].isin(teams)]

print(matches_data.shape)

matches_data = matches_data[matches_data['dl_applied'] == 0]

matches_data = matches_data[['match_id','city','winner','total_runs']]

deliveries_data = matches_data.merge(deliveries,on='match_id')

deliveries_data = deliveries_data[deliveries_data['inning'] == 2]

print(deliveries_data)

deliveries_data['current_score'] = deliveries_data.groupby('match_id').cumsum()['total_runs_y']

deliveries_data['runs_left'] = deliveries_data['total_runs_x'] - deliveries_data['current_score']

deliveries_data['balls_left'] = 126 - (deliveries_data['over']*6 + deliveries_data['ball'])

print(deliveries_data)

deliveries_data['player_dismissed'] = deliveries_data['player_dismissed'].fillna("0")
deliveries_data['player_dismissed'] = deliveries_data['player_dismissed'].apply(lambda x:x if x == "0" else "1")
deliveries_data['player_dismissed'] = deliveries_data['player_dismissed'].astype('int')
wickets = deliveries_data.groupby('match_id').cumsum()['player_dismissed'].values
deliveries_data['wickets'] = 10 - wickets
print(deliveries_data.head())

deliveries_data['crr'] = (deliveries_data['current_score']*6)/(120 - deliveries_data['balls_left'])

deliveries_data['rrr'] = (deliveries_data['runs_left']*6)/deliveries_data['balls_left']

def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0
deliveries_data['result'] = deliveries_data.apply(result,axis=1)

last_data = deliveries_data[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]

last_data = last_data.sample(last_data.shape[0])

print(last_data.head())

last_data.dropna(inplace=True)

last_data = last_data[last_data['balls_left'] != 0]

X = last_data.iloc[:,:-1]
y = last_data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

print(X_train)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])],remainder='passthrough')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[('step1',trf),('step2',LogisticRegression(solver='liblinear'))])
pipe.fit(X_train,y_train)

Pipeline(steps=[('step1',ColumnTransformer(remainder='passthrough',transformers=[('trf',OneHotEncoder(drop='first',sparse=False),['batting_team','bowling_team', 'city'])])),('step2', LogisticRegression(solver='liblinear'))])

y_pred = pipe.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

print(pipe.predict_proba(X_test)[10])

def match_data(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))

def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_data = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_data = temp_data[temp_data['balls_left'] != 0]
    result = pipe.predict_proba(temp_data)
    temp_data['lose'] = np.round(result.T[0]*100,1)
    temp_data['win'] = np.round(result.T[1]*100,1)
    temp_data['end_of_over'] = range(1,temp_data.shape[0]+1)
    
    target = temp_data['total_runs_x'].values[0]
    runs = list(temp_data['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_data['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_data['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_data['wickets_in_over'] = (nw - w)[0:temp_data.shape[0]]
    
    print("Target-",target)
    temp_data = temp_data[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_data,target
    
temp_data,target = match_progression(deliveries_data,45,pipe)
print(temp_data)

import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_data['end_of_over'],temp_data['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_data['end_of_over'],temp_data['win'],color='#00a65a',linewidth=4)
plt.plot(temp_data['end_of_over'],temp_data['lose'],color='red',linewidth=4)
plt.bar(temp_data['end_of_over'],temp_data['runs_after_over'])
plt.title('Target-' + str(target))

print(teams)

print(deliveries_data['city'].unique())

import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))
