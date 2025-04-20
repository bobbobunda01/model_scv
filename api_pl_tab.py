#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:59:03 2025

@author: bobunda
"""

from joblib import load
from pydantic import BaseModel
from flask import Flask, jsonify, request
from typing import List
import numpy as np
import pandas as pd

app = Flask(__name__)

# Modèle Pydantic pour une entrée
class MatchInput(BaseModel):
    HomeTeam: str
    AwayTeam: str
    comp: str

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
            
            if comp=='pl':
                
                # Chargement des données de la Première league
                
                # Chargement des données historiques
                hi=pd.read_csv('pl_match_03_2025_hist_net.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_pl_smote.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
                
            elif comp=='sa':
                # Chargement des données historiques
                hi=pd.read_csv('sa_25_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_sa_smote.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
            
            elif comp=='lg':
                # Chargement des données historiques
                hi=pd.read_csv('lg_25_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_lg_smote.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
                
            elif comp=='bl':
                # Chargement des données historiques
                hi=pd.read_csv('bl_25_01_04_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_bl_smote.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
            
            elif comp=='fl':
                # Chargement des données historiques
                hi=pd.read_csv('fl_25_14_04_2025.csv')
                hi.drop('Unnamed: 0', axis=1, inplace=True)
                hi['Date']=pd.to_datetime(hi['Date'])
                df=hi

                # chargement de paramètres de standardisation des données de pretraitement
                scale=pd.read_csv('dp_scale_fl_smote.csv')
                scale.set_index('Unnamed: 0', inplace=True)
                ds_scale=scale
                
            #perf=df[df['HomeTeam']==home].sort_values(by='Date', ascending=False).head(1)
            
            cols_home=['HomeTeam','FTHG',  'HTGS', 'HTGC', 'HTHG', 'HHGS' ,'HHGC','HTP',
               'HM1' ,'HM2', 'HM3', 'HM4', 'HM5', 'HTFormPts', 'HTWinStreak3', 
               'HTWinStreak5', 'HTLossStreak3','HTGD', 'HHGD', 'Date']
            cols_away=['AwayTeam','FTAG', 'ATGS', 'ATGC', 'HTAG', 'AHGS' ,'AHGC','ATP',
               'AM1' ,'AM2', 'AM3', 'AM4', 'AM5', 'ATFormPts', 'ATWinStreak3', 
               'ATWinStreak5', 'ATLossStreak3','ATGD', 'AHGD', 'Date']
            #DiffPts=HTP-ATP
            #DiffFormPts=HTFormPts-ATFormPts
            
            #df_home=df.loc[df['HomeTeam']==home,cols_home].sort_values(by='Date', ascending=False).drop('Date', axis=1).head(1)
            #df_away=df.loc[df['AwayTeam']==away,cols_away].sort_values(by='Date', ascending=False).drop('Date', axis=1).head(1)
            #df_home=df_home.reset_index()
            #df_away=df_away.reset_index()
            
            date_hh=df.loc[df['HomeTeam']==home,['Date']].sort_values(by='Date', ascending=False).head(1)
            date_hw=df.loc[df['AwayTeam']==home,['Date']].sort_values(by='Date', ascending=False).head(1)
            df_home=pd.DataFrame()
            df_away=pd.DataFrame()
            
            if date_hh['Date'].values>date_hw['Date'].values:
                df_home=df.loc[df['HomeTeam']==home,cols_home].sort_values(by='Date', ascending=False).head(1)
                print(home)
                print("Home is his last form")
            else:
                print(home)
                print("away is his last form")
                df_home=df.loc[df['AwayTeam']==home,cols_away].sort_values(by='Date', ascending=False).head(1)
                df_home.columns=cols_home
            
            date_hh=df.loc[df['HomeTeam']==away,['Date']].sort_values(by='Date', ascending=False).head(1)
            date_hw=df.loc[df['AwayTeam']==away,['Date']].sort_values(by='Date', ascending=False).head(1)
            
            if date_hh['Date'].values>date_hw['Date'].values:
                print(away)
                print("Home is his last form")
                df_away=hi.loc[hi['HomeTeam']==away,cols_home].sort_values(by='Date', ascending=False).head(1)
                df_away.columns=cols_away
            else:
                print(away)
                print("away is his last form")
                df_away=hi.loc[hi['AwayTeam']==away,cols_away].sort_values(by='Date', ascending=False).head(1)
            
            perf_home=df_home['HM1']+df_home['HM2']+df_home['HM3']+df_home['HM4']+df_home['HM5']
            perf_away=df_away['AM1']+df_away['AM2']+df_away['AM3']+df_away['AM4']+df_away['AM5']
            
            df_home=df_home.reset_index()
            df_away=df_away.reset_index()
            
            df_home_away=pd.concat([df_home, df_away], axis=1)
            df_home_away=df_home_away.drop('index', axis=1)
            
            df_home_away['DiffPts']=df_home_away['HTP']-df_home_away['ATP']
            df_home_away['DiffFormPts']=df_home_away['HTFormPts']-df_home_away['ATFormPts']
            df_home_away=df_home_away.drop('Date', axis=1)
            df_home_away=standardize_user_input(df_home_away, ds_scale)
            
            # Transformation des données

            x_t=df_home_away.drop(['HomeTeam','AwayTeam'], axis=1)
            x_t = pd.get_dummies(x_t, columns=['HM1','HM2','HM3','HM4','HM5','AM1', 'AM2', 'AM3', 'AM4', 'AM5'])
            x_t.replace({True:1, False:0}, inplace=True)
            
            #bon modèle
            if comp=='pl':
                modele=load('scv_model.joblib')
            
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
            #y_proba= xgboost_model.predict_proba(x_t)
            
            
            # compilation des resultats dans un dictionnaire
            
            
            # Adaptation des sorties
            result = match.dict()
            result.update({
                '5_dern_perf_home': np.array(perf_home).item(),
                '5_dern_perf_away': np.array(perf_away).item(),
                'resultat': int(Y_pred)
            })
            all_results.append(result)

        return jsonify({'Resultats': all_results})

    except Exception as e:
        return jsonify({'Erreur': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)