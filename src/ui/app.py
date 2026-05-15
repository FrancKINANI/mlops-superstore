import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

import os

# Configuration de la page
st.set_page_config(
    page_title="Superstore Profitability Predictor",
    page_icon="💰",
    layout="wide"
)

# Configuration de l'API - Utilise localhost par défaut, mais permet d'être surchargé par ENV (pour Docker)
API_URL = os.getenv("API_URL", "http://localhost:8000")

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# Titre de l'application
st.title("💰 Superstore Profitability Predictor")
st.markdown("""
Cette application permet de prédire si une transaction de vente sera **rentable** ou non.
Elle communique avec l'API FastAPI qui utilise un modèle de Machine Learning entraîné sur le dataset Superstore.
""")

# Sidebar pour la santé de l'API et les informations du modèle
with st.sidebar:
    st.header("État du Système")
    health = check_api_health()
    if health:
        st.success("✅ API Connectée")
        st.json(health)
    else:
        st.error("❌ API Déconnectée")
        st.info(f"Assurez-vous que l'API est lancée sur {API_URL}")
        if st.button("Réessayer la connexion"):
            st.rerun()

    st.divider()
    st.header("À propos")
    st.info("Projet MLOps - Prédiction de Rentabilité")

# Formulaire de saisie des données
st.header("📝 Saisie de la Transaction")

col1, col2, col3 = st.columns(3)

with col1:
    sales = st.number_input("Chiffre de vente ($)", min_value=0.1, value=150.0, step=10.0)
    quantity = st.number_input("Quantité", min_value=1, value=2, step=1)
    discount = st.slider("Remise (Discount)", min_value=0.0, max_value=0.8, value=0.1, step=0.05)
    unit_price = st.number_input("Prix unitaire ($)", min_value=0.1, value=sales/quantity if quantity > 0 else 0.0)

with col2:
    ship_mode = st.selectbox("Mode d'expédition", 
                           ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
    segment = st.selectbox("Segment client", 
                         ['Consumer', 'Corporate', 'Home Office'])
    region = st.selectbox("Région", 
                        ['West', 'East', 'Central', 'South'])
    category = st.selectbox("Catégorie", 
                          ['Office Supplies', 'Furniture', 'Technology'])

with col3:
    sub_category = st.selectbox("Sous-catégorie", 
                              ['Paper', 'Binders', 'Art', 'Phones', 'Storage', 'Appliances', 
                               'Accessories', 'Chairs', 'Furnishings', 'Labels', 'Envelopes', 
                               'Fasteners', 'Supplies', 'Bookcases', 'Tables', 'Machines', 'Copiers'])
    
    # Dates et délais
    order_date = st.date_input("Date de commande", datetime.now())
    shipping_delay = st.number_input("Délai de livraison (jours)", min_value=0, value=4, step=1)

# Calcul des features temporelles
order_month = order_date.month
order_quarter = (order_date.month - 1) // 3 + 1
order_dayofweek = order_date.weekday()

# Préparation de la requête
payload = {
    "Sales": sales,
    "Quantity": quantity,
    "Discount": discount,
    "Ship_Mode": ship_mode,
    "Segment": segment,
    "Region": region,
    "Category": category,
    "Sub_Category": sub_category,
    "order_month": order_month,
    "order_quarter": order_quarter,
    "order_dayofweek": order_dayofweek,
    "shipping_delay": shipping_delay,
    "unit_price": unit_price
}

st.divider()

# Bouton de prédiction
if st.button("🚀 Prédire la rentabilité", use_container_width=True):
    with st.spinner("Analyse en cours..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Affichage du résultat
                st.header("📊 Résultat de l'Analyse")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if result['prediction'] == 1:
                        st.success(f"### {result['label']}")
                    else:
                        st.error(f"### {result['label']}")
                    
                    st.metric("Confiance du modèle", f"{result['confidence']*100:.2f}%")
                
                with res_col2:
                    st.write("**Détails techniques :**")
                    st.write(f"- Modèle : `{result['model_name']}`")
                    st.write(f"- Stage : `{result['model_stage']}`")
                    st.write(f"- Probabilité : `{result['probability']}`")
                
                # Visualisation de la probabilité
                st.progress(result['probability'], text=f"Probabilité de la classe : {result['probability']:.4f}")
                
            else:
                st.error(f"Erreur API ({response.status_code})")
                st.json(response.json())
        except Exception as e:
            st.error(f"Erreur lors de la communication avec l'API : {e}")

# Section Debug/JSON
with st.expander("Voir les données envoyées (JSON)"):
    st.json(payload)
