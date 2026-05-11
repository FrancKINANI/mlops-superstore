"""
Tests unitaires et d'intégration pour l'API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Ajouter le projet root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app, get_model
from src.config import config


@pytest.fixture
def client():
    """Fixture: client FastAPI pour les tests."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests des endpoints de santé."""
    
    def test_root_endpoint(self, client):
        """Test endpoint racine."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "preprocessor_loaded" in data


class TestPredictionEndpoint:
    """Tests du endpoint de prédiction."""
    
    @pytest.fixture
    def valid_transaction(self):
        """Fixture: transaction valide."""
        return {
            "Sales": 150.0,
            "Quantity": 2,
            "Discount": 0.1,
            "Ship_Mode": "Standard Class",
            "Segment": "Consumer",
            "Region": "East",
            "Category": "Office Supplies",
            "Sub_Category": "Paper",
            "order_month": 5,
            "order_quarter": 2,
            "order_dayofweek": 2,
            "shipping_delay": 4,
            "unit_price": 75.0
        }
    
    def test_valid_prediction(self, client, valid_transaction):
        """Test prédiction valide."""
        response = client.post("/predict", json=valid_transaction)
        
        # Vérifier le statut
        assert response.status_code == 200, f"Erreur: {response.text}"
        
        # Vérifier la structure
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "label" in data
        assert "model_name" in data
        assert "model_stage" in data
        assert "timestamp" in data
        assert "confidence" in data
        
        # Vérifier les valeurs
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert 0 <= data["confidence"] <= 1
        assert data["label"] in ["Rentable 💰", "Non rentable ❌"]
    
    def test_missing_field(self, client, valid_transaction):
        """Test avec champ manquant."""
        del valid_transaction["Sales"]
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_invalid_discount(self, client, valid_transaction):
        """Test avec discount invalide."""
        valid_transaction["Discount"] = 1.5  # > 1
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 422
    
    def test_negative_quantity(self, client, valid_transaction):
        """Test avec quantité négative."""
        valid_transaction["Quantity"] = -1
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 422
    
    def test_zero_sales(self, client, valid_transaction):
        """Test avec ventes nulles."""
        valid_transaction["Sales"] = 0
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 422
    
    def test_invalid_month(self, client, valid_transaction):
        """Test avec mois invalide."""
        valid_transaction["order_month"] = 13
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 422
    
    def test_invalid_quarter(self, client, valid_transaction):
        """Test avec trimestre invalide."""
        valid_transaction["order_quarter"] = 5
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Tests du endpoint de prédiction par batch."""
    
    @pytest.fixture
    def valid_transactions(self):
        """Fixture: liste de transactions valides."""
        base = {
            "Sales": 150.0,
            "Quantity": 2,
            "Discount": 0.1,
            "Ship_Mode": "Standard Class",
            "Segment": "Consumer",
            "Region": "East",
            "Category": "Office Supplies",
            "Sub_Category": "Paper",
            "order_month": 5,
            "order_quarter": 2,
            "order_dayofweek": 2,
            "shipping_delay": 4,
            "unit_price": 75.0
        }
        return [base, base.copy(), base.copy()]
    
    def test_batch_prediction(self, client, valid_transactions):
        """Test prédiction par batch."""
        response = client.post("/batch_predict", json=valid_transactions)
        assert response.status_code == 200
        
        data = response.json()
        assert "total" in data
        assert "successful" in data
        assert "failed" in data
        assert "predictions" in data
        assert data["total"] == len(valid_transactions)
    
    def test_empty_batch(self, client):
        """Test batch vide."""
        response = client.post("/batch_predict", json=[])
        assert response.status_code == 400
    
    def test_batch_too_large(self, client):
        """Test batch trop large."""
        base = {
            "Sales": 150.0,
            "Quantity": 2,
            "Discount": 0.1,
            "Ship_Mode": "Standard Class",
            "Segment": "Consumer",
            "Region": "East",
            "Category": "Office Supplies",
            "Sub_Category": "Paper",
            "order_month": 5,
            "order_quarter": 2,
            "order_dayofweek": 2,
            "shipping_delay": 4,
            "unit_price": 75.0
        }
        large_batch = [base] * 1001  # > 1000
        response = client.post("/batch_predict", json=large_batch)
        assert response.status_code == 400


class TestModelInfoEndpoint:
    """Tests du endpoint d'info sur le modèle."""
    
    def test_model_info(self, client):
        """Test endpoint model info."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "stage" in data
        assert "path" in data
        assert "model_type" in data
        assert "cache_enabled" in data
        assert "monitoring_enabled" in data


class TestDocumentation:
    """Tests de la documentation."""
    
    def test_docs_available(self, client):
        """Test que la documentation est accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema(self, client):
        """Test que le schéma OpenAPI est accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/predict" in data["paths"]


class TestErrorHandling:
    """Tests de gestion d'erreurs."""
    
    def test_invalid_json(self, client):
        """Test JSON invalide."""
        response = client.post("/predict", content="invalid json", 
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 422
    
    def test_method_not_allowed(self, client):
        """Test méthode HTTP non autorisée."""
        response = client.get("/predict")  # GET au lieu de POST
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
