#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 19:39:21 2025

@author: bobunda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 10:02:57 2024

@author: bobunda
"""

import requests
import json

# lien d'accès

url_base='http://127.0.0.1:5000'
#url_base='https://model-scv.onrender.com'
#https://model-scv.onrender.com

#url_base='https://model-scv.onrender.com'

# Test de point d'accès d'accueil
#reponse=requests.get(f"{url_base}/")
##
#print("reponse de point d'accès:", reponse.text) 

# Données d'exemple pour la prédiction

data={

    "matches": [
        
        {
            "HomeTeam": "Genoa",
            "AwayTeam": "Atalanta",
            "comp": "sa",
            "odds_home":4.34,
            "odds_draw":3.84,
            "odds_away":1.79
        },
        {
            "HomeTeam": "Cagliari",
            "AwayTeam": "Venezia FC",
            "comp": "sa",
            "odds_home":2.59,
            "odds_draw":3.2,
            "odds_away":1.75
        },
        {
            "HomeTeam": "Monza",
            "AwayTeam": "Empoli",
            "comp": "sa",
            "odds_home":4.87,
            "odds_draw":3.6,
            "odds_away":1.5
        },
        {
            "HomeTeam": "Fiorentina",
            "AwayTeam": "Bologna",
            "comp": "sa",
            "odds_home":2.76,
            "odds_draw":3.11,
            "odds_away":2.72
        },
        
        {
            "HomeTeam": "Inter",
            "AwayTeam": "Lazio",
            "comp": "sa",
            "odds_home":1.64,
            "odds_draw":4.11,
            "odds_away":5.03
        }
    ]
}

# Envoi de la requête POST
response = requests.post(f"{url_base}/predire/pl", json=data)
# Affichage de la réponse
response_data=response.json()
formatted_json = json.dumps(response_data, indent=2, ensure_ascii=False)  # Indentation de 2 espaces
print(formatted_json)