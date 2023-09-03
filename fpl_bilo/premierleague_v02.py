# Linear Programming with PL
import requests
import json
import pandas as pd
import os
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
os.add_dll_directory(r"C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex2210.dll")
# import docplex.mp.models
from docplex.mp.model import Model
import numpy as np
import math
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.tree import ExtraTreeClassifier
import logging, warnings
from tqdm import tqdm
from data_modules import get_data
import itertools

class PremierLeague:

    def __init__(self):

        print('\n-*-*- PREMIER LEAGUE OPTIMISER -*-*-')

        self.position_mapping = {1: 'G', 2: 'D', 3: 'M', 4: 'F'}
        self.team_mapping = None # TODO
        
        self.player_data = get_data.PLData().get_prem_league_player_data()
        self.player_data.to_clipboard()

    # OPTIMISE TEAM
    def optimise_team(self, budget=1000, players=15, squad=None, select_available=True, optimise_on='total_points', remove_player_id=None):

        if select_available:
            self.player_data = self.player_data.loc[self.player_data['status'] == 'a']

        self.player_data.reset_index(inplace=True, drop=True)

        # TEAM RULES
        if squad is None:
            squad = {'G': 2, 'D': 5, 'M': 5, 'F': 3, 'same_team': 3}

        if remove_player_id is not None:
            for i in remove_player_id:
                self.player_data = self.player_data[self.player_data['code'] != i]
                self.player_data.reset_index(inplace=True, drop=True)

        # UNIQUE LIST OF TEAMS
        teams = list(self.player_data['team'].drop_duplicates())

        # CRITERIA FOR SELECTION IS MAXIMISE POINTS PER GAME
        self.player_data['optimise_var'] = self.player_data[optimise_on].astype(float)

        # SETUP MODEL PARAMETERS
        m = Model(name='TEST')

        # Objective Function
        variables_dict = {}
        total_cost = 0
        team_players = 0
        total_revenue = 0

        goalkeepers = 0
        defenders = 0
        midfielders = 0
        forwards = 0

        team_dict = {}
        for t in teams:
            team_dict[t] = 0

        for i in range(len(self.player_data)):

            player_id = self.player_data.loc[i:i, 'name'][i]
            player_cost = self.player_data.loc[i:i, 'now_cost'][i]
            player_points = self.player_data.loc[i:i, 'optimise_var'][i]
            player_type = self.player_data.loc[i:i, 'element_type'][i]
            team_type = self.player_data.loc[i:i, 'team'][i]

            variable = m.binary_var(name=f'{player_id}')
            variables_dict[i] = variable
            m.add_constraint(variables_dict[i] <= 1)
            total_cost += variable * player_cost
            total_revenue += variable * player_points
            team_players += variable
            team_dict[team_type] += variable

            # NUMBER OF PLAYER TYPES
            if player_type == 1:  # goalkeeper
                goalkeepers += variable
            if player_type == 2:  # goalkeeper
                defenders += variable
            if player_type == 3:  # goalkeeper
                midfielders += variable
            if player_type == 4:  # goalkeeper
                forwards += variable

            # MAX [ X ] PLAYERS PER TEAM
            for t in teams:
                m.add_constraint(team_dict[t] <= squad['same_team'])

        total_players_in_team = players
        total_budget = budget

        m.add_constraint(total_cost <= total_budget)  # WE CAN'T EXCEED OUR BUDGET
        m.add_constraint(team_players == total_players_in_team)  # WE MUST HAVE 15 PLAYERS

        # SQUAD CONSTRAINTS
        m.add_constraint(goalkeepers <= squad['G'])
        m.add_constraint(defenders <= squad['D'])
        m.add_constraint(midfielders <= squad['M'])
        m.add_constraint(forwards <= squad['F'])

        # OBJECTIVE FUNCTION, MAXIMISE REVENUE
        m.maximize(total_revenue)
        solved = m.solve()

        self.optimised_team = solved.as_df()

        self.optimised_team = self.optimised_team.merge(self.player_data[['name', 'element_type', 'team', 'optimise_var', 'chance_of_playing_next_round', 'id', 'now_cost']], on='name', how='left')
        self.optimised_team.sort_values(by=['element_type'], ascending=True, inplace=True, ignore_index=True)
        pd.set_option('display.max_columns', 10)

        # self.optimised_team['team'] = self.optimised_team['team'].map(self.team_mapping) # TODO
        self.optimised_team['element_type'] = self.optimised_team['element_type'].map(self.position_mapping)
        self.optimised_team = self.optimised_team.drop(columns=['value'])

        print(f'total score: {self.optimised_team["optimise_var"].sum()}, Cost: {self.optimised_team["now_cost"].sum()}')
        print(self.optimised_team[['name']])

        return self.optimised_team


    def get_sub_combinations(self):

        combs = itertools.combinations('GGDDDDDMMMMMFFF', 11)
        combs = [i for i in set(combs) if i.count('G') == 1 and i.count('F') >= 1 and i.count('D') >= 3]

        squad = {'G': 2, 'D': 5, 'M': 5, 'F': 3}

        subs = []

        for i in combs:

            merged_list = []

            for j in squad.keys():

                if not squad[j] - i.count(j) == 0:
                    merged_list.append((squad[j] - i.count(j)) * [j])

            result = tuple(sum(merged_list, []))

            squad_arg = {'G': 2, 'D': 5, 'M': 5, 'F': 3, 'same_team': 3}
            budget_arg = 1000

            for j in result:

                temp = self.player_data[['now_cost', 'element_type']].copy()

                temp = temp.groupby(by=['element_type']).min()

                for key, value in self.position_mapping.items():
                    if value == j:

                        budget_arg -= temp.iloc[key -1, 0]
                        # print(f'minus {temp.iloc[key -1, 0]}')
                        squad_arg[j] -= 1

            print(result)
            # print(squad_arg, budget_arg)
            self.optimise_team(budget=budget_arg, players=11, squad=squad_arg, remove_player_id=[78830])
            print()


            subs.append(result)

        return subs

    # SELECT CAPTAIN FROM TEAM
    def select_captain(self):

        logging.disable()
        warnings.filterwarnings('ignore')

        self.all_detailed_data = pd.read_csv(r'C:\Users\alexa\Desktop\detailed_player_data.csv')
        self.all_detailed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.all_detailed_data.sort_values(by='kickoff_time', ascending=True, inplace=True)
        self.all_detailed_data.dropna(axis=0, inplace=True)
        self.all_detailed_data = self.all_detailed_data.loc[self.all_detailed_data['minutes'] > 0]

        for i in list(self.all_detailed_data['element'].drop_duplicates()):
            temp = self.all_detailed_data.loc[self.all_detailed_data['element'] == i]
            temp = temp['total_points'].to_numpy()
            try:
                gradient = np.gradient(temp)
            except:
                gradient = 0

            mean = np.mean(temp)

            self.all_detailed_data.loc[self.all_detailed_data['element'] == i, 'mean_optimise_var'] = mean
            self.all_detailed_data.loc[self.all_detailed_data['element'] == i, 'season_form'] = gradient

        self.all_detailed_data['log_value'] =  self.all_detailed_data['value'].apply(lambda x: math.log10(x))
        self.all_detailed_data['log_game_difficulty'] =  self.all_detailed_data['game_difficulty'].apply(lambda x: math.log10(x))
        self.all_detailed_data['log_opposition_game_difficulty'] =  self.all_detailed_data['opposition_game_difficulty'].apply(lambda x: math.log10(x))
        self.all_detailed_data['log_total_points'] =  self.all_detailed_data['total_points'].apply(lambda x: math.log10(x) if x > 0 else 0)

        pd.get_dummies(self.all_detailed_data, columns=['was_home'])
        self.all_detailed_data['home'] = 0
        self.all_detailed_data.loc[self.all_detailed_data['was_home'], 'home'] = 1

        X = self.all_detailed_data[['log_game_difficulty', 'home', 'log_value', 'mean_optimise_var', 'season_form']]
        y = self.all_detailed_data['total_points'].to_numpy()

        models = [LogisticRegression(max_iter=600, solver='newton-cholesky'), LogisticRegression(max_iter=600,
                                                                                                 solver='newton-cg'), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=500), GradientBoostingClassifier(),
                  xgb.XGBClassifier(), LinearRegression(), SGDClassifier(), ExtraTreeClassifier()]

        # models = [LogisticRegression(max_iter=5000, solver='newton-cholesky'), LogisticRegression(max_iter=5000, solver='newton-cg'), LogisticRegression(max_iter=5000, solver='liblinear'), LogisticRegression(max_iter=5000, solver='sag'), LogisticRegression(max_iter=5000, solver='saga')]
        best = 100

        for model in models:

            for _ in tqdm(range(100)):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                model.fit(X_train, y_train)

                prediction = model.predict(X_test)
                mae = mean_absolute_error(prediction, y_test)

                if mae < best:
                    best = mae
                    print(model)
                    print(f"MAE: {mae:.3f}")
                    self.model = model
                    # pickle.dump(self.models, open('captain_model.pkl', 'wb'))

        print(f"BEST RMSE: {best:.3f}")

        return self.model


    @staticmethod
    def find_key_by_value(dictionary, value):
        for key, val in dictionary.items():
            if val == value:
                return key
        return None  # Return None if the value is not found

if __name__ == "__main__":

    PremierLeague().get_sub_combinations()
    # PremierLeague().optimise_team(budget=1000, players=15, squad = {'G': 2, 'D': 5, 'M': 5, 'F': 3, 'same_team': 3})




