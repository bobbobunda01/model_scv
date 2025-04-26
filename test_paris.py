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

#print("reponse de point d'accès:", reponse.text)

# Données d'exemple pour la prédiction

data={

    "matches": [
        
        
        {
            "HomeTeam": "Wolves",
            "AwayTeam": "Tottenham",
            "comp": "pl"
            
        },

        {
            "HomeTeam": "Newcastle",
            "AwayTeam": "Man United",
            "comp": "pl"
            
        }
    ]
}

# Envoi de la requête POST
response = requests.post(f"{url_base}/predire/pl", json=data)
# Affichage de la réponse
response_data=response.json()
formatted_json = json.dumps(response_data, indent=2, ensure_ascii=False)  # Indentation de 2 espaces
print(formatted_json)