import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

# load data
games_data = pd.read_csv('./data/enhanced_games.csv')

# feature matrix
features = ['HomeTeamForm', 'HomeTeamGoals', 'HomeTeamRating',
            'AwayTeamForm', 'AwayTeamGoals', 'AwayTeamRating']
X = games_data[features]

# target variable
y = games_data['FTR'].map({'H': 1, 'A': 0, 'D': 2})

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')

# train model
grid_search.fit(X_train, y_train)

# save model
with open('./models/trained_model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)