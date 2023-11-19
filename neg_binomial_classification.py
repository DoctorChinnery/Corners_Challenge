import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pymc3 as pm


# Read in the data edited and formatted by build_train_data.py:
train_15 = pd.read_csv(r'train (15) processed.csv')
train_15['Over_10_Corners'] = np.where(train_15['Total_Corners'] > 11.5, 1, 0)
train_15['Over_10_Corners'] = train_15['Over_10_Corners'].astype('category')

# Select training features:
X = train_15[['HomeTeamId', 'AwayTeamId', 'LeagueId', 'Home_last_3_goals', 'Away_last_3_goals', 'Home_last_3_corners',
              'Away_last_3_corners', 'League_Average_Corners']]
y = train_15['Over_10_Corners']

# Create a validation and test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
test_data = pd.concat([y_test, X_test], axis=1)
train_data = pd.concat([y_train, X_train], axis=1)
print(X_train.columns.values)
#with pm.Model(coords={"predictors": })


