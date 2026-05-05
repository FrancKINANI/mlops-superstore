# start_mlflow_tunnel.py
import os
import subprocess
import time
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError

# Lance MLflow en arrière-plan
mlflow_process = subprocess.Popen([
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5000"
])

time.sleep(3)

# Configure ngrok auth token from environment if available
auth_token = os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("PYNGROK_AUTHTOKEN")
if auth_token:
    ngrok.set_auth_token(auth_token)

# Ouvre le tunnel ngrok
try:
    tunnel = ngrok.connect(5000)
    print(f"\n MLflow Tracking URI pour Colab :")
    print(f" {tunnel.public_url}")
except PyngrokNgrokError as err:
    print("\nErreur ngrok : impossible d'ouvrir le tunnel ngrok.")
    print(f"{err}\n")
    print("Assure-toi d'avoir un compte ngrok vérifié et un authtoken configuré.")
    print("1) Crée un compte : https://dashboard.ngrok.com/signup")
    print("2) Récupère ton authtoken : https://dashboard.ngrok.com/get-started/your-authtoken")
    print("3) Configure l'authtoken dans l'environnement : export NGROK_AUTHTOKEN=...\n")
    print("Ou exécute 'ngrok authtoken <token>' si ngrok est installé localement.")

print(f"\n MLflow UI local : http://localhost:5000")
print("\n Laisse ce terminal ouvert pendant ta session Colab.")