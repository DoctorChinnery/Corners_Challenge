import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Read in the data edited and formatted by build_train_data.py:
train_15 = pd.read_csv(r'train (15) processed.csv')
train_15['Over_10_Corners'] = np.where(train_15['Total_Corners'] > 10, 1, 0)
train_15['Over_10_Corners'] = train_15['Over_10_Corners'].astype('category')

# Select training features:
X = train_15[['HomeTeamId', 'AwayTeamId', 'LeagueId', 'Home_last_3_goals', 'Away_last_3_goals', 'Home_last_3_corners',
              'Away_last_3_corners', 'League_Average_Corners']]
y = train_15['Over_10_Corners']

# Batching
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

train_dataset = cb.Pool(data=X_train, label=y_train)
test_dataset = cb.Pool(data=X_test, label=y_test)
cv_dataset = cb.cv(cb.Pool(data=X, label=y), params={'loss_function': cb.metrics.Logloss()})
# Construct model:
#iterations=1000, depth=6, learning_rate=0.1,
model = cb.CatBoostClassifier(loss_function='Logloss', verbose=True, learning_rate=0.1)
model.fit(train_dataset)
predictions = model.predict(test_dataset)
pred_prob = model.predict_proba(test_dataset)
print(cv_dataset)
print('Cross validation accuracy = {}%'.format(100*np.round(np.max(cv_dataset['test-Accuracy-mean']))))
print("1-fold accuracy = {}%".format(np.round(accuracy_score(y_test, predictions),3) * 100))