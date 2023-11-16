import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
data = pd.read_csv(r"C:\Users\jackb\Downloads\train (15).csv")
df = pd.DataFrame(data)
corners = df[['Home_Corners', 'Away_Corners']]
total_corners = corners['Home_Corners'].to_numpy() + corners['Away_Corners'].to_numpy()
mean_corners = np.mean(total_corners)
corner_number = np.arange(max(total_corners) + 1)
corner_freq = [0.0] * (max(total_corners) + 1)
total_no = len(total_corners)

def last_3_mean_home(dataframe,teamID,matchID):
    #Find column of the specific match
    match_column = dataframe.index[dataframe['MatchId'] == matchID][0]
    #Get last three games
    data = dataframe.loc[dataframe['HomeTeamId'] == teamID, 'Home_Corners'].loc[:match_column]
    if len(data) < 4:
        mean = data.iloc[:len(data) - 1].mean()
    else:
        mean = data.tail(3).mean()
    return mean

def last_3_mean_away(dataframe,teamID,matchID):
    #Find column of the specific match
    match_column = dataframe.index[dataframe['MatchId'] == matchID][0]
    #Get last three games
    data = dataframe.loc[dataframe['AwayTeamId'] == teamID, 'Away_Corners'].loc[:match_column]
    if len(data) < 4:
        mean = data.iloc[:len(data) - 1].mean()
    else:
        mean = data.tail(3).mean()
    return mean
