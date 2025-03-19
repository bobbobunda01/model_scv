#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:45:42 2024

@author: bobunda
"""

from joblib import load
from pydantic import BaseModel
from flask import Flask, jsonify, request


import numpy as np
import pandas as pd

# Application de la plateforme flask
# création de l'instance flask de l'application

app=Flask(__name__)

# création de l'objet

class request_body(BaseModel):
    
    HomeTeam:str
    AwayTeam:str



@app.route('/', methods=["GET"])
def Acceuil():
    return jsonify({'Message':'Soyez le bienvenu'})


# Chargement du modele

xgboost_model =load('svm_modele.joblib')

def data_df(df, model):
    
    for feature in model.feature_names_in_:
        if feature not in df.columns:
            df[feature] = False  # or np.nan, depending on your use case
    # Reorder columns to match training data
    df = df[model.feature_names_in_]
    df.replace({True:1, False:0}, inplace=True)
    return df

df=pd.read_csv('pl_match_03_2025_hist_net.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# choix de l'équipe

@app.route('/predire/pl', methods=["POST"])
def prediction():
    if not request.json:
        return jsonify({'Erreur':'Aucun fichier Joson fourni'}), 400
    
    try:
        # Extraction des données d'entrée en json
        donnees= request_body(**request.json) 
        donnees_df=pd.DataFrame([donnees.dict()]) # conversion en DataFrame
        home=np.array(donnees_df.HomeTeam.values).item()
        
        perf=df[df['HomeTeam']==home].sort_values(by='Date', ascending=False).head(1)
        
        # Transformation des données

        x_t=perf.drop(['FTR', 'HTR', 'Date', 'HomeTeam'	,'AwayTeam'], axis=1)
        x_t = pd.get_dummies(x_t, columns=['HM1','HM2','HM3','HM4','HM5','AM1', 'AM2', 'AM3', 'AM4', 'AM5'])
        x_t.replace({True:1, False:0}, inplace=True)
        
        #prediction 
        x_t=data_df(x_t, xgboost_model)

        Y_pred = xgboost_model.predict(x_t)
        #y_proba= xgboost_model.predict_proba(x_t)
        
        
        # compilation des resultats dans un dictionnaire
        resul=donnees.dict()
        resul['WIN']=int(Y_pred)
        #resul['proba']=float(y_proba[1])
        
        # Renvoie les résultats sous forme de Json
        return jsonify({'Resultats':resul})
    
    except Exception as e:
        
        # Renvoie d'une reponse d'erreur
        return jsonify({'Erreur':str(e)}), 400  




if __name__=='__main__':
    app.run(debug=True)