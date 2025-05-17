#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:59:03 2025

@author: bobunda
"""

import datetime
import json
from joblib import load
from pydantic import BaseModel
from flask import Flask, jsonify, request
from typing import List
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Modèle Pydantic pour une entrée
class MatchInput(BaseModel):
    HomeTeam: str
    AwayTeam: str
    comp: str
    odds_home:float
    odds_draw:float
    odds_away:float
    

# Modèle pour recevoir un tableau d'entrées
class RequestBody(BaseModel):
    matches: List[MatchInput]  # Accepte un tableau de 4 entrées

@app.route('/', methods=["GET"])
def Accueil():
    return jsonify({'Message': 'Bienvenue sur l\'API de prédiction de matchs'})



def data_df(df, model):
    
    for feature in model.feature_names_in_:
        if feature not in df.columns:
            df[feature] = False  # or np.nan, depending on your use case
    # Reorder columns to match training data
    df = df[model.feature_names_in_]
    df.replace({True:1, False:0}, inplace=True)
    return df

def standardize_user_input(input_user, ds_scale):
    
    input_standardized = input_user.copy()

    # Appliquer la standardisation pour chaque colonne
    for column in input_standardized.select_dtypes(exclude='object').columns:
        mean = ds_scale.at['mean', column]  # Récupérer la moyenne
        std = ds_scale.at['std', column]    # Récupérer l'écart-type
        input_standardized[column] = (input_standardized[column] - mean) / std

    return input_standardized
# Fonction pour logger chaque prédiction dans logs/logs.jsonl
def log_prediction(prediction):
    log_data = {
        "request_date": datetime.datetime.utcnow().isoformat(),
        #"input": data,
        "prediction": prediction
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/logs.jsonl", "a") as f:
        f.write(json.dumps(log_data) + "\n")
        
def form(d_plf):

    def get_points(result):
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0
    
    def get_form_points(string):
        sum = 0
        for letter in string:
            sum += get_points(letter)
        return sum

    d_plf['HTFormPtsStr'] = d_plf['HM1'] + d_plf['HM2'] + d_plf['HM3'] + d_plf['HM4'] + d_plf['HM5']
    d_plf['ATFormPtsStr'] = d_plf['AM1'] + d_plf['AM2'] + d_plf['AM3'] + d_plf['AM4'] + d_plf['AM5']

    d_plf['HTFormPts'] = d_plf['HTFormPtsStr'].apply(get_form_points)
    d_plf['ATFormPts'] = d_plf['ATFormPtsStr'].apply(get_form_points)

    # Identify Win/Loss Streaks if any.
    def get_3game_ws(string):
        if string[-3:] == 'WWW':
            return 1
        else:
            return 0

    def get_5game_ws(string):
        if string == 'WWWWW':
            return 1
        else:
            return 0

    def get_3game_ls(string):
        if string[-3:] == 'LLL':
            return 1
        else:
            return 0

    def get_5game_ls(string):
        if string == 'LLLLL':
            return 1
        else:
            return 0

    d_plf['HTWinStreak3'] = d_plf['HTFormPtsStr'].apply(get_3game_ws)
    d_plf['HTWinStreak5'] = d_plf['HTFormPtsStr'].apply(get_5game_ws)
    d_plf['HTLossStreak3'] = d_plf['HTFormPtsStr'].apply(get_3game_ls)
    d_plf['HTLossStreak5'] = d_plf['HTFormPtsStr'].apply(get_5game_ls)

    d_plf['ATWinStreak3'] = d_plf['ATFormPtsStr'].apply(get_3game_ws)
    d_plf['ATWinStreak5'] = d_plf['ATFormPtsStr'].apply(get_5game_ws)
    d_plf['ATLossStreak3'] = d_plf['ATFormPtsStr'].apply(get_3game_ls)
    d_plf['ATLossStreak5'] = d_plf['ATFormPtsStr'].apply(get_5game_ls)
    return d_plf

def df_data(sa_25, home, away):
    cols_home=['HomeTeam','FTHG', 'FTAG','HTHG', 'HTAG', 'HTGS', 'HTGC','HHGS' ,'HHGC','HTP','HM1' ,'HM2', 'HM3', 'HM4', 'HM5', 'Date']
    cols_away=['AwayTeam','FTHG','FTAG','HTHG', 'HTAG' ,'ATGS', 'ATGC', 'AHGS' ,'AHGC','ATP','AM1' ,'AM2', 'AM3', 'AM4', 'AM5','Date']
    
    #HOME
    df_home=pd.DataFrame()
    df_away=pd.DataFrame()
    ### Home
    date_hh=sa_25.loc[sa_25['HomeTeam']==home,['Date']].sort_values(by='Date', ascending=False).head(1)
    date_hw=sa_25.loc[sa_25['AwayTeam']==home,['Date']].sort_values(by='Date', ascending=False).head(1)
    if date_hh['Date'].values>date_hw['Date'].values:
        df_home=sa_25.loc[sa_25['HomeTeam']==home,cols_home].sort_values(by='Date', ascending=False).head(1)
        df_home['HTGS']=df_home['HTGS']+df_home['FTHG']
        df_home['HTGC']=df_home['HTGC']+df_home['FTAG']
        df_home['HHGS']=df_home['HHGS']+df_home['HTHG']
        df_home['HHGC']=df_home['HHGC']+df_home['HTAG']
        if df_home['FTHG'].values>df_home['FTAG'].values:
            df_home['HTP']=df_home['HTP']+3
            a='W'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
        elif df_home['FTHG'].values==df_home['FTAG'].values:
            df_home['HTP']=df_home['HTP']+1
            a='D'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
        else:
            a='L'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
    else:#
        df_home=sa_25.loc[sa_25['AwayTeam']==home,cols_away].sort_values(by='Date', ascending=False).head(1)
        df_home.columns=cols_home
        #df_home['HTHG']=pd.to_numeric(df['HTHG'], errors='coerce')
        df_home['HTGS']=df_home['HTGS']+df_home['FTAG']
        df_home['HTGC']=df_home['HTGC']+df_home['FTHG']
        df_home['HHGS']=df_home['HHGS']+df_home['HTAG']
        df_home['HHGC']=df_home['HHGC']+df_home['HTHG']
        if df_home['FTAG'].values>df_home['FTHG'].values:
            df_home['HTP']=df_home['HTP']+3
            a='W'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
            df_home['FTHG']=df_home['FTAG']
        elif df_home['FTAG'].values==df_home['FTHG'].values:
            df_home['HTP']=df_home['HTP']+1
            a='D'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
            df_home['FTHG']=df_home['FTAG']
        else:
            a='L'
            df_home['HM5']=df_home['HM4']
            df_home['HM4']=df_home['HM3']
            df_home['HM3']=df_home['HM2']
            df_home['HM2']=df_home['HM1']
            df_home['HM1']=a
            df_home['FTHG']=df_home['FTAG']
    #df_home['FTHG']=round((df_home['HTGS']/32),0)
    #df_home=df_home.drop(['FTAG','HTAG'], axis=1)
    #AWAY
    date_hh=sa_25.loc[sa_25['HomeTeam']==away,['Date']].sort_values(by='Date', ascending=False).head(1)
    date_hw=sa_25.loc[sa_25['AwayTeam']==away,['Date']].sort_values(by='Date', ascending=False).head(1)
    if date_hh['Date'].values>date_hw['Date'].values:
        df_away=sa_25.loc[sa_25['HomeTeam']==away,cols_home].sort_values(by='Date', ascending=False).head(1)
        df_away.columns=cols_away
        df_away['ATGS']=df_away['ATGS']+df_away['FTHG']
        df_away['ATGC']=df_away['ATGC']+df_away['FTAG']
        df_away['AHGS']=df_away['AHGS']+df_away['HTHG']
        df_away['AHGC']=df_away['AHGC']+df_away['HTAG']

        if df_away['FTHG'].values>df_away['FTAG'].values:
            df_away['ATP']=df_away['ATP']+3
            a='W'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
            df_away['FTAG']=df_home['FTHG']
        elif df_away['FTHG'].values==df_away['FTAG'].values:
            df_away['ATP']=df_away['ATP']+1
            a='D'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
            df_away['FTAG']=df_home['FTHG']
        else:
            a='L'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
            df_away['FTAG']=df_home['FTHG']
    else:#
        df_away=sa_25.loc[sa_25['AwayTeam']==away,cols_away].sort_values(by='Date', ascending=False).head(1)
        df_away['ATGS']=df_away['ATGS']+df_away['FTAG']
        df_away['ATGC']=df_away['ATGC']+df_away['FTHG']
        df_away['AHGS']=df_away['AHGS']+df_away['HTAG']
        df_away['AHGC']=df_away['AHGC']+df_away['HTHG']

        if df_away['FTAG'].values>df_away['FTHG'].values:
            df_away['ATP']=df_away['ATP']+3
            a='W'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
        elif df_away['FTHG'].values==df_away['FTAG'].values:
            df_away['ATP']=df_away['ATP']+1
            a='D'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
        else:
            a='L'
            df_away['AM5']=df_away['AM4']
            df_away['AM4']=df_away['AM3']
            df_away['AM3']=df_away['AM2']
            df_away['AM2']=df_away['AM1']
            df_away['AM1']=a
    #df_away['FTAG']=round((df_away['ATGS']/32),0)
    #df_away=df_away.drop(['FTHG', 'HTHG'], axis=1)

    df_home=df_home.reset_index()
    df_away=df_away.reset_index()
    df_home_away=pd.concat([df_home, df_away], axis=1)
    df_home_away=df_home_away.drop('index', axis=1)
    df_home_away
    df_home_away=form(df_home_away)
    df_home_away['DiffPts']=df_home_away['HTP']-df_home_away['ATP']
    df_home_away['DiffFormPts']=df_home_away['HTFormPts']-df_home_away['ATFormPts']

    #d_plf
    # Get Goal Difference
    df_home_away['HTGD'] = df_home_away['HTGS'] - df_home_away['HTGC']
    df_home_away['ATGD'] = df_home_away['ATGS'] - df_home_away['ATGC']
    df_home_away['HHGD'] = df_home_away['HHGS'] - df_home_away['HHGC']
    df_home_away['AHGD'] = df_home_away['AHGS'] - df_home_away['AHGC']

    # Diff in points
    df_home_away['DiffPts'] = df_home_away['HTP'] - df_home_away['ATP']
    df_home_away['DiffFormPts'] = df_home_away['HTFormPts'] - df_home_away['ATFormPts']
    cols = ['HTGD','ATGD', 'HHGD', 'AHGD','DiffPts','DiffFormPts','HTP','ATP']
    for col in cols:
        df_home_away[col] = df_home_away[col] /33
    return df_home_away, df_home, df_away
def get_team_stats(df, team, n=5):
    """
    Récupère les stats des N derniers matchs pour une équipe (home + away confondus)
    """
    team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values(by='Date', ascending=False).head(n)
    
    goals_for = []
    goals_against = []
    goal_diff = []
    points = []
    results = []
    
    for _, row in team_matches.iterrows():
        if row['HomeTeam'] == team:
            gf = row['FTHG']
            ga = row['FTAG']
            gd = gf - ga
            if row['FTR'] == 'H':
                pts = 3
                res = 'W'
            elif row['FTR'] == 'D':
                pts = 1
                res = 'D'
            else:
                pts = 0
                res = 'L'
        else:  # Away
            gf = row['FTAG']
            ga = row['FTHG']
            gd = gf - ga
            if row['FTR'] == 'A':
                pts = 3
                res = 'W'
            elif row['FTR'] == 'D':
                pts = 1
                res = 'D'
            else:
                pts = 0
                res = 'L'
        
        goals_for.append(gf)
        goals_against.append(ga)
        goal_diff.append(gd)
        points.append(pts)
        results.append(res)
    
    stats = {
        'Avg_GF_Last{}_Games'.format(n): np.mean(goals_for) if goals_for else 0,
        'Avg_GA_Last{}_Games'.format(n): np.mean(goals_against) if goals_against else 0,
        'PPM_Last{}_Games'.format(n): np.mean(points) if points else 0,
        'Win_Rate_Last{}_Games'.format(n): results.count('W') / n if n > 0 else 0,
        'Win_Streak_Current': count_streak(results, 'W'),
        'Lose_Streak_Current': count_streak(results, 'L'),
        'Att_Def_Ratio_Last{}_Games'.format(n): (np.mean(goals_for) / np.mean(goals_against)) if np.mean(goals_against) > 0 else np.nan,
        'Strength_Index': (0.5 * (results.count('W') / n)) + (0.3 * (np.mean(goals_for) - np.mean(goals_against))) + (0.2 * np.mean(points)),
        'GD_Last{}_Avg'.format(n): np.mean(goal_diff) if goal_diff else 0
    }
    
    return stats

def count_streak(results, target='W'):
    """
    Compte le streak courant (victoires/défaites consécutives au début de la série)
    """
    streak = 0
    for r in results:
        if r == target:
            streak += 1
        else:
            break
    return streak

def prepare_features_for_next_match(df, home_team, away_team, odds_home, odds_draw, odds_away):
    """
    Prépare les features prospectifs pour un match à venir
    """
    home_stats = get_team_stats(df, home_team)
    away_stats = get_team_stats(df, away_team)
    
    # Combine en une seule ligne de features
    features = {}
    
    # Add home team stats
    for k, v in home_stats.items():
        features['Home_' + k] = v
        
    # Add away team stats
    for k, v in away_stats.items():
        features['Away_' + k] = v
    
    # Add the odds
    features['B365H'] = odds_home
    features['B365D'] = odds_draw
    features['B365A'] = odds_away
    features['Odds_HA_Diff'] = odds_home - odds_away
    features['Odds_HD_Diff'] = odds_home - odds_draw
    
    return features
# definition des variables 
df=pd.DataFrame()
ds_scale=pd.DataFrame()

@app.route('/predire/pl', methods=["POST"])
def prediction():
    if not request.json:
        return jsonify({'Erreur': 'Aucun fichier JSON fourni'}), 400
    
    try:
        # Extraction des 4 entrées
        body = RequestBody(**request.json)
        all_results = []

        for match in body.matches:
            # Traitement pour chaque match
            donnees_df = pd.DataFrame([match.dict()])
            
            home=np.array(donnees_df.HomeTeam.values).item()
            away=np.array(donnees_df.AwayTeam.values).item()
            comp=np.array(donnees_df.comp.values).item()
            odds_h = donnees_df["odds_home"].values[0]
            odds_d = donnees_df["odds_draw"].values[0]
            odds_a = donnees_df["odds_away"].values[0]
            
            if comp=='pl':
                
                # Chargement des données de la Première league
                
                # Chargement des données historiques
                hi=pd.read_csv('pl_match_03_2025_hist_net.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_pl.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
                
            elif comp=='sa':
                # Chargement des données historiques
                hi=pd.read_csv('sa_25_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_sa.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
            
            elif comp=='lg':
                # Chargement des données historiques
                hi=pd.read_csv('lg_25_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_lg.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
                
            elif comp=='bl':
                # Chargement des données historiques
                hi=pd.read_csv('bl_25_25_04_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_bl.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
            
            elif comp=='fl':
                # Chargement des données historiques
                hi=pd.read_csv('fl_26_04_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_fl.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
            
            df_home_away, df_home, df_away=df_data(df, home, away)
            
            df_home_away = df_home_away.loc[:, ~df_home_away.columns.duplicated(keep='first')]
            
            df_test=prepare_features_for_next_match(df, home, away, odds_h, odds_d, odds_a)
            new_df = pd.DataFrame([df_test])
            
            df_home_away = pd.concat([df_home_away, new_df], axis=1)
            
            # Probabilités implicites à partir des cotes Bet365
            df_home_away['Prob_H'] = 1 / df_home_away['B365H']
            df_home_away['Prob_D'] = 1 / df_home_away['B365D']
            df_home_away['Prob_A'] = 1 / df_home_away['B365A']

            # Normalisation des probabilités
            df_home_away['Prob_Total'] = df_home_away['Prob_H'] + df_home_away['Prob_D'] + df_home_away['Prob_A']
            df_home_away['Prob_H'] /= df_home_away['Prob_Total']
            df_home_away['Prob_D'] /= df_home_away['Prob_Total']
            df_home_away['Prob_A'] /= df_home_away['Prob_Total']

            # Deltas entre probabilités
            df_home_away['Delta_Prob_HD'] = df_home_away['Prob_H'] - df_home_away['Prob_D']
            df_home_away['Delta_Prob_HA'] = df_home_away['Prob_H'] - df_home_away['Prob_A']

            # Volatilité du marché
            df_home_away['Volatility_Index'] = abs(df_home_away['Prob_H'] - df_home_away['Prob_A'])
            
            df_home_away.rename(columns={'Away_Strength_Index':'Strength_Index_Away'}, inplace=True)
            
            df_home_away=df_home_away[['HomeTeam','AwayTeam','HTP','HTFormPts','Prob_H', 
                           'B365H','B365D', 'B365A','ATP', 'ATFormPts', 'Delta_Prob_HD','Delta_Prob_HA', 
                           'DiffFormPts','DiffPts','HM1', 'AM1','HM2','AM2','HM3','AM3', 'HM4','AM4','HM5', 'AM5',
                           'Odds_HD_Diff', 'Odds_HA_Diff', 'Strength_Index_Away', 'Away_GD_Last5_Avg',
                           'ATWinStreak5', 'ATGD', 'AHGD', 'B365D', 'Prob_D']]
            
            df_home_away = df_home_away.loc[:, ~df_home_away.columns.duplicated(keep='first')]
            
            
            perf_home=df_home['HM1']+df_home['HM2']+df_home['HM3']+df_home['HM4']+df_home['HM5']
            perf_away=df_away['AM1']+df_away['AM2']+df_away['AM3']+df_away['AM4']+df_away['AM5']
            
            df_home_away=standardize_user_input(df_home_away, ds_scale)
            
            # Transformation des données

            x_t=df_home_away.drop(['HomeTeam','AwayTeam'], axis=1)
            x_t = pd.get_dummies(x_t, columns=['HM1','HM2','HM3','HM4','HM5','AM1', 'AM2', 'AM3', 'AM4', 'AM5'])
            x_t.replace({True:1, False:0}, inplace=True)
            
            #bon modèle
            if comp=='pl':
                modele=load('lg_softmax_pl.joblib')
            
            #bon modèle    
            elif comp=='sa':
                modele=load('lg_softmax_sa.joblib')
            ##bon modele
            elif comp=='lg':
                modele=load('lg_softmax_lg.joblib')
            ##BON MODÈLE
            elif comp=='bl':
                modele=load('lg_softmax_bl.joblib')
                
            ## Bon modèle
            elif comp=='fl':
                modele=load('lg_softmax_fl.joblib')
            
            
            #prediction 
            x_t=data_df(x_t, modele)

            Y_pred = modele.predict(x_t)
            y_proba= modele.predict_proba(x_t)
            
            
            # compilation des resultats dans un dictionnaire
            
            
            # Adaptation des sorties
            result = match.dict()
            result.update({
                '5_dern_perf_home': np.array(perf_home).item(),
                '5_dern_perf_away': np.array(perf_away).item(),
                'resultat': int(Y_pred),
                'proba_0':str(round(y_proba[0][0]*100,0))+'%',
                'proba_1':str(round(y_proba[0][1]*100,0))+'%',
                'proba_2':str(round(y_proba[0][2]*100,0))+'%'
            })
            all_results.append(result)
            # Log l'entrée + les prédictions
            log_prediction(all_results)
        return jsonify({'Resultats': all_results})

    except Exception as e:
        return jsonify({'Erreur': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)