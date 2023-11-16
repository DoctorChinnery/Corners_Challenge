import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Load data, and take out the basic features for use:
train_15 = pd.read_csv(r"C:\Users\jackb\Downloads\train (15).csv")
train_15['Total_Corners'] = train_15['Home_Corners'] + train_15['Away_Corners']
train_15['MatchId'] = train_15['MatchId']
train_15['HomeTeamId'] = train_15['HomeTeamId'].astype('category')
train_15['AwayTeamId'] = train_15['AwayTeamId'].astype('category')
train_15['LeagueId'] = train_15['LeagueId'].astype('category')
train_15['Over_10_Corners'] = np.where(train_15['Total_Corners'] > 10, 1, 0)
train_15['Over_10_Corners'] = train_15['Over_10_Corners'].astype('category')


# Find last three game home and away goals:
def last_3_mean_goals_home(dataframe, teamID, matchID):
    # Find column of the specific match
    match_column = dataframe.index[dataframe['MatchId'] == matchID][0]
    # Get last three games
    data = dataframe.loc[dataframe['HomeTeamId'] == teamID, 'Home_Corners'].loc[:match_column]
    if len(data) < 4:
        mean = data.iloc[:len(data) - 1].mean()
    else:
        mean = data.tail(3).mean()
    return mean


def last_3_mean_away(dataframe, teamID, matchID):
    # Find column of the specific match
    match_column = dataframe.index[dataframe['MatchId'] == matchID][0]
    # Get last three games
    data = dataframe.loc[dataframe['AwayTeamId'] == teamID, 'Away_Corners'].loc[:match_column]
    if len(data) < 4:
        mean = data.iloc[:len(data) - 1].mean()
    else:
        mean = data.tail(3).mean()
    return mean


train_15['Home_last_3_goals'] = train_15.apply(
    lambda x: last_3_mean_goals_home(train_15, x['HomeTeamId'], x['MatchId']), axis=1)
train_15['Away_last_3_goals'] = train_15.apply(
    lambda x: last_3_mean_away(train_15, x['AwayTeamId'], x['MatchId']), axis=1)

# Find last three game home and away corners:
def last_3_mean_corners_home(dataframe, teamID, matchID):
    # Find column of the specific match
    match_column = dataframe.index[dataframe['MatchId'] == matchID][0]
    # Get last three games
    data = dataframe.loc[dataframe['HomeTeamId'] == teamID, 'Home_Corners'].loc[:match_column]
    if len(data) < 4:
        mean = data.iloc[:len(data) - 1].mean()
    else:
        mean = data.tail(3).mean()
    return mean


def last_3_mean_corners_away(dataframe, teamID, matchID):
    # Find column of the specific match
    match_column = dataframe.index[dataframe['MatchId'] == matchID][0]
    # Get last three games
    data = dataframe.loc[dataframe['AwayTeamId'] == teamID, 'Away_Corners'].loc[:match_column]
    if len(data) < 4:
        mean = data.iloc[:len(data) - 1].mean()
    else:
        mean = data.tail(3).mean()
    return mean


train_15['Home_last_3_corners'] = train_15.apply(
    lambda x: last_3_mean_corners_home(train_15, x['HomeTeamId'], x['MatchId']), axis=1)
train_15['Away_last_3_corners'] = train_15.apply(
    lambda x: last_3_mean_corners_away(train_15, x['AwayTeamId'], x['MatchId']), axis=1)


# Find league average corners:
def league_average_corners(dataframe, leagueID, matchID):
    # Find column of the specific match
    match_column = dataframe.index[dataframe['MatchId'] == matchID][0]
    # Get all games in league
    data = dataframe.loc[dataframe['LeagueId'] == leagueID, 'Total_Corners'].loc[:match_column]
    mean = data.mean()
    return mean


train_15['League_Average_Corners'] = train_15.apply(
    lambda x: league_average_corners(train_15, x['LeagueId'], x['MatchId']), axis=1)

# Select training features:
X = train_15[['HomeTeamId', 'AwayTeamId', 'LeagueId', 'Home_last_3_goals', 'Away_last_3_goals', 'Home_last_3_corners',
              'Away_last_3_corners', 'League_Average_Corners']]
y = train_15['Over_10_Corners']
# Batching
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

# Construct and train model:
model = xgb.XGBClassifier(objective='binary:logistic', enable_categorical=True)
model = GridSearchCV(model, {'max_depth': [2, 4, 6, 8], 'n_estimators': [50, 100, 200, 400]}, scoring='accuracy', verbose=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
pred_prob = model.predict_proba(X_test)
print(pred_prob)
print(min(pred_prob[0]))
print("accuracy::", accuracy_score(y_test, predictions))
