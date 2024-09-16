import pandas as pd

# team form
def calculate_team_form(data, n_games=5):
    team_form = {}

    def update_form(row, team, result):
        if team not in team_form:
            team_form[team] = []
        team_form[team].append(result)
        if len(team_form[team]) > n_games:
            team_form[team].pop(0)
        return sum(team_form[team]) / len(team_form[team]) if len(team_form[team]) > 0 else 0

    data['HomeTeamForm'] = data.apply(lambda row: update_form(row, row['HomeTeam'], 1 if row['FTHG'] > row['FTAG'] else (0.5 if row['FTHG'] == row['FTAG'] else 0)), axis=1)
    data['AwayTeamForm'] = data.apply(lambda row: update_form(row, row['AwayTeam'], 1 if row['FTAG'] > row['FTHG'] else (0.5 if row['FTHG'] == row['FTHG'] else 0)), axis=1)

    return data

# average goals
def calculate_team_goals(data):
    data['HomeTeamGoals'] = data.groupby('HomeTeam')['FTHG'].transform('mean')
    data['AwayTeamGoals'] = data.groupby('AwayTeam')['FTAG'].transform('mean')
    return data

# average player rating
def add_team_ratings(data, players_data):
    team_avg_rating = players_data.groupby('Team-name')['Rating'].mean()

    data['HomeTeamRating'] = data['HomeTeam'].map(team_avg_rating)
    data['AwayTeamRating'] = data['AwayTeam'].map(team_avg_rating)

    return data

# load data
games_data = pd.read_csv('./data/la-liga-results-1995-2020.csv')
players_data = pd.read_csv('./data/player-status-2022-2023.csv')

# calculate form, goals, ratings
games_data = calculate_team_form(games_data)
games_data = calculate_team_goals(games_data)
games_data = add_team_ratings(games_data, players_data)

# save enhanced data
games_data.to_csv('./data/enhanced_games.csv', index=False)