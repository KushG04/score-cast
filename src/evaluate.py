import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# load model
def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

# main
if __name__ == "__main__":
    model = load_model('./models/trained_model.pkl')
    test_data = pd.read_csv('./data/enhanced_games.csv')

    features = ['HomeTeamForm', 'HomeTeamGoals', 'HomeTeamRating',
                'AwayTeamForm', 'AwayTeamGoals', 'AwayTeamRating']
    
    X_test = test_data[features]
    y_test = test_data['FTR'].map({'H': 1, 'A': 0, 'D': 2})
    
    evaluate_model(model, X_test, y_test)