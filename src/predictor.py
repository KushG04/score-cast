import pickle
import pandas as pd

# valid teams
valid_teams = [
    "Barcelona", "Real Madrid", "Atletico Madrid", "Valencia", "Sevilla",
    "Villarreal", "Real Sociedad", "Real Betis", "Athletic Club", "Espanyol",
    "Real Valladolid", "Getafe", "Osasuna", "Celta de Vigo", "Las Palmas", "Alaves",
    "Leganes", "Mallorca", "Girona", "Rayo Vallecano"
]

# check team validity
def get_valid_team(prompt):
    team_name = input(prompt)
    while team_name not in valid_teams:
        print(f"'{team_name}' is not a valid La Liga team. Please try again.")
        team_name = input(prompt)
    return team_name

# load model
def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# team form
def get_team_form(team_name, games_data, num_games=5):
    team_games = games_data[(games_data['HomeTeam'] == team_name) | (games_data['AwayTeam'] == team_name)]
    recent_games = team_games.tail(num_games)
    
    form = 0
    for _, game in recent_games.iterrows():
        if game['HomeTeam'] == team_name:
            if game['FTR'] == 'H':
                form += 1
            elif game['FTR'] == 'D':
                form += 0.5
        elif game['AwayTeam'] == team_name:
            if game['FTR'] == 'A':
                form += 1
            elif game['FTR'] == 'D':
                form += 0.5
    return form / num_games

# average goals
def get_team_goals(team_name, games_data):
    home_goals = games_data[games_data['HomeTeam'] == team_name]['FTHG'].mean()
    away_goals = games_data[games_data['AwayTeam'] == team_name]['FTAG'].mean()
    return (home_goals + away_goals) / 2

# team rating
def get_team_rating(team_name, players_data):
    team_rating = players_data[players_data['Team-name'] == team_name]['Rating'].mean()
    return team_rating

# prediction using model
def make_prediction(model, input_features, home_team, away_team):
    feature_names = ['HomeTeamForm', 'HomeTeamGoals', 'HomeTeamRating', 
                     'AwayTeamForm', 'AwayTeamGoals', 'AwayTeamRating']
    input_df = pd.DataFrame([input_features], columns=feature_names)
    
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        return f"The predicted winner is the home team: {home_team}"
    elif prediction == 0:
        return f"The predicted winner is the away team: {away_team}"
    else:
        return "The predicted result is a draw."

# main
if __name__ == "__main__":
    model = load_model('./models/trained_model.pkl')
    games_data = pd.read_csv('./data/la-liga-results-1995-2020.csv')
    players_data = pd.read_csv('./data/player-status-2022-2023.csv')
    
    home_team = get_valid_team("Home Team: ")
    away_team = get_valid_team("Away Team: ")
    
    home_team_form = get_team_form(home_team, games_data)
    home_team_goals = get_team_goals(home_team, games_data)
    home_team_rating = get_team_rating(home_team, players_data)
    
    away_team_form = get_team_form(away_team, games_data)
    away_team_goals = get_team_goals(away_team, games_data)
    away_team_rating = get_team_rating(away_team, players_data)
    
    input_features = [home_team_form, home_team_goals, home_team_rating,
                      away_team_form, away_team_goals, away_team_rating]
    
    result = make_prediction(model, input_features, home_team, away_team)
    print(result)