import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from patsy import dmatrices
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
# Read in the data edited and formatted by build_train_data.py:
train_15 = pd.read_csv(r'train (15) processed.csv')
train_15['Over_10_Corners'] = np.where(train_15['Total_Corners'] > 11.5, 1, 0)
train_15['Over_10_Corners'] = train_15['Over_10_Corners'].astype('category')

# Select training features:
X = train_15[['HomeTeamId', 'AwayTeamId', 'LeagueId', 'Home_last_3_goals', 'Away_last_3_goals', 'Home_last_3_corners',
              'Away_last_3_corners', 'League_Average_Corners']]
y = train_15['Total_Corners']

# Create a validation and test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
test_data = pd.concat([y_test, X_test], axis=1)
train_data = pd.concat([y_train, X_train], axis=1)

# Regression model:
expression = """Total_Corners ~ HomeTeamId + AwayTeamId + LeagueId + Home_last_3_goals + Away_last_3_goals +
                Home_last_3_corners + Away_last_3_corners + League_Average_Corners"""
y_train,X_train = dmatrices(expression, train_data, return_type='dataframe')
y_test, X_test = dmatrices(expression, test_data, return_type='dataframe')

# Train to get series of lambdas:
poisson_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
print(poisson_results.summary())


# Now need to find the value of alpha:
alpha_train = X_train.copy()
alpha_train['lamda'] = poisson_results.mu
alpha_train['y'] = y_train['Total_Corners']
alpha_train['alpha'] = alpha_train.apply(lambda x: ((x['y'] - x['lamda'])**2) - x['y'] / x['lamda'], axis=1)

# Now find the value of alpha:
expression_OLS = """alpha ~ lamda - 1"""
aux_ols_results = smf.ols(expression_OLS, alpha_train).fit()

# Using the value of alpha, construct the negative binomial 2 model:
nb2_results = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=aux_ols_results.params[0])).fit()
print(nb2_results.summary())
nb2_predictions = nb2_results.predict(X_test)
print(len(nb2_predictions))
print(len(y_test))

print(mean_squared_error(y_test, nb2_predictions)**0.5)




