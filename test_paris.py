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

# lien d'accès

url_base='http://127.0.0.1:5000'
#url_base='https://model-scv.onrender.com'
#https://model-scv.onrender.com

# Test de point d'accès d'accueil
reponse=requests.get(f"{url_base}/")

#print("reponse de point d'accès:", reponse.text)

# Données d'exemple pour la prédiction

data={

    "matches": [
        {
            "HomeTeam": "Liverpool",
            "AwayTeam": "Southampton",
            "comp": "pl"
        },
        {
            "HomeTeam": "Chelsea",
            "AwayTeam": "Leicester",
            "comp": "pl"
        },
        {
            "HomeTeam": "Man United",
            "AwayTeam": "Tottenham",
            "comp": "pl"
        },
        {
            "HomeTeam": "Everton",
            "AwayTeam": "Bournemouth",
            "comp": "pl"
        }
    ]
}

# Envoi de la requête POST
response = requests.post(f"{url_base}/predire/pl", json=data)
print(response.text) 
# Affichage de la réponse
print(response.json())