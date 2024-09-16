import pandas as pd

# load data
def load_data(filepath):
    return pd.read_csv(filepath)

# clean data
def clean_data(data):
    data.fillna(0, inplace=True)
    return data

# main
if __name__ == "__main__":
    games = load_data('./data/la-liga-results-1995-2020.csv')
    players = load_data('./data/player-status-2022-2023.csv')
    games = clean_data(games)
    players = clean_data(players)
    
    games.to_csv('./data/cleaned_games.csv', index=False)
    players.to_csv('./data/cleaned_players.csv', index=False)