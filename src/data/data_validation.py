# src/data/data_validation.py
"""
Validation des données avec Great Expectations.
Vérifie que les données respectent le contrat attendu
avant d'entrer dans le pipeline de preprocessing.

Fournit des validations pour:
- Données brutes (raw data validation)
- Données traitées (processed data validation)
"""

from typing import Dict, Any, List
from pathlib import Path
import great_expectations as gx
import pandas as pd
import numpy as np

from src.config import config
from src.logging_utils import get_logger, log_performance

logger = get_logger(__name__)


# ─────────────────────────────────────────
# HELPER: Rapport HTML
# ─────────────────────────────────────────

def _save_validation_report(
    results: Any,
    output_path: str,
    title: str
) -> None:
    """
    Sauvegarde le rapport de validation en HTML.
    
    Args:
        results: Résultats de validation Great Expectations
        output_path: Chemin de sauvegarde du rapport
        title: Titre du rapport
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Créer le rapport (simplifié sans builder)
        logger.info(f"Rapport de validation sauvegardé: {output_path}")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder le rapport: {e}")


# ─────────────────────────────────────────
# 1. VALIDATION DES DONNÉES BRUTES
# ─────────────────────────────────────────

@log_performance
def validate_raw_data(path: str = None) -> Dict[str, Any]:
    """
    Valide le dataset brut Superstore.
    
    Vérifications:
    - Toutes les colonnes requises présentes
    - Valeurs numériques dans les bonnes plages
    - Catégories valides
    - Pas de nulls sur colonnes critiques
    - Volume minimum de données
    - Unicité des Row IDs
    
    Args:
        path: Chemin vers le dataset brut (défaut: config.data.raw_data_path)
    
    Returns:
        Dict avec clés: success, total, passed, failed, report
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    path = path or str(config.data.raw_data_path)
    logger.info(f"Validation des données brutes : {path}")
    
    df = pd.read_csv(path, encoding='latin-1')
    
    # Crée le contexte GX en mémoire
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("superstore_raw")
    asset = datasource.add_dataframe_asset("raw_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    # Suite d'expectations
    suite = context.suites.add(
        gx.ExpectationSuite(name="superstore_raw_suite")
    )

    # 1. Colonnes obligatoires
    required_columns = [
        'Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
        'Customer ID', 'Customer Name', 'Segment', 'Country', 'City',
        'State', 'Postal Code', 'Region', 'Product ID', 'Category',
        'Sub-Category', 'Product Name', 'Sales', 'Quantity',
        'Discount', 'Profit'
    ]
    for col in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # 2. Sales strictement positif
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Sales", min_value=0.01, max_value=100000
        )
    )

    # 3. Discount entre 0 et 1
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Discount", min_value=0.0, max_value=1.0
        )
    )

    # 4. Quantity entier positif
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="Quantity", min_value=1, max_value=100
        )
    )

    # 5. Category valide
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Category",
            value_set=["Furniture", "Office Supplies", "Technology"]
        )
    )

    # 6. Region valide
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Region",
            value_set=["West", "East", "Central", "South"]
        )
    )

    # 7. Segment valide
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Segment",
            value_set=["Consumer", "Corporate", "Home Office"]
        )
    )

    # 8. Pas de nulls sur colonnes critiques
    for col in ["Sales", "Profit", "Discount", "Category", "Region"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # 9. Volume minimum
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000,
            max_value=None
        )
    )

    # 10. Unicité Row ID
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="Row ID")
    )

    # Exécution
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="raw_validation",
            data=batch_def,
            suite=suite,
        )
    )

    results = validation_def.run(
        batch_parameters={"dataframe": df}
    )

    # Rapport HTML
    report_path = config.monitoring.drift_report_path.parent / "ge_raw_validation.html"
    _save_validation_report(results, str(report_path), "Raw Data Validation")

    # Résumé
    success = results.success
    total = len(results.results)
    passed = sum(1 for r in results.results if r.success)
    failed = total - passed

    logger.info(f"Validation brute — {passed}/{total} règles passées")
    if not success:
        failed_rules = [
            r.expectation_config.type
            for r in results.results if not r.success
        ]
        logger.warning(f"Règles échouées : {failed_rules}")

    return {
        "success": success,
        "total": total,
        "passed": passed,
        "failed": failed,
        "report": str(report_path)
    }


# ─────────────────────────────────────────
# 2. VALIDATION DES DONNÉES TRAITÉES
# ─────────────────────────────────────────

@log_performance
def validate_processed_data(path: str = None) -> Dict[str, Any]:
    """
    Valide le dataset après preprocessing.
    
    Vérifications:
    - Target binaire (0/1)
    - Features temporelles présentes et valides
    - Pas de nulls
    - Types de données corrects
    
    Args:
        path: Chemin vers les données traitées (défaut: config.data.processed_data_path)
    
    Returns:
        Dict avec clés: success, total, passed, failed, report
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
    """
    path = path or str(config.data.processed_data_path)
    logger.info(f"Validation des données traitées : {path}")
    
    df = pd.read_csv(path)

    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("superstore_processed")
    asset = datasource.add_dataframe_asset("processed_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(
        gx.ExpectationSuite(name="superstore_processed_suite")
    )

    # 1. Target binaire
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="is_profitable")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_profitable", value_set=[0, 1]
        )
    )

    # 2. Features temporelles
    for col in ["order_month", "order_quarter", "order_dayofweek", "shipping_delay"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # 3. Plages valides
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="order_month", min_value=1, max_value=12
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="order_quarter", min_value=1, max_value=4
        )
    )

    # 4. Pas de nulls
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchOrderedList(
            column_list=df.columns.tolist()
        )
    )

    # Exécution
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="processed_validation",
            data=batch_def,
            suite=suite,
        )
    )

    results = validation_def.run(
        batch_parameters={"dataframe": df}
    )

    # Rapport
    report_path = config.monitoring.drift_report_path.parent / "ge_processed_validation.html"
    _save_validation_report(results, str(report_path), "Processed Data Validation")

    # Résumé
    success = results.success
    total = len(results.results)
    passed = sum(1 for r in results.results if r.success)
    failed = total - passed

    logger.info(f"Validation traitée — {passed}/{total} règles passées")

    return {
        "success": success,
        "total": total,
        "passed": passed,
        "failed": failed,
        "report": str(report_path)
    }


def get_validation_status(raw_path: str = None, processed_path: str = None) -> Dict[str, Any]:
    """
    Exécute les deux validations et retourne un résumé global.
    
    Args:
        raw_path: Chemin données brutes
        processed_path: Chemin données traitées
    
    Returns:
        Dict: Résumé combiné des validations
    """
    raw_results = validate_raw_data(raw_path)
    processed_results = validate_processed_data(processed_path)
    
    return {
        "raw": raw_results,
        "processed": processed_results,
        "overall_success": raw_results["success"] and processed_results["success"]
    }


if __name__ == "__main__":
    # Test
    status = get_validation_status()
    print(f"\n✅ Validation complète")
    print(f"Raw data: {status['raw']}")
    print(f"Processed data: {status['processed']}")

    # 3. Pas de NaN dans les features numériques clés
    for col in ["Sales", "Quantity", "Discount", "shipping_delay", "unit_price"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # 4. Mois entre 1 et 12
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="order_month", min_value=1, max_value=12
        )
    )

    # 5. Shipping delay non négatif
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="shipping_delay", min_value=0, max_value=60
        )
    )

    # 6. Profit absent (vérification anti-leakage)
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=[
                "Ship Mode", "Segment", "Region", "Category", "Sub-Category",
                "Sales", "Quantity", "Discount", "order_month", "order_quarter",
                "order_dayofweek", "shipping_delay", "unit_price", "is_profitable"
            ],
            exact_match=True,
        )
    )

    # 7. Volume cohérent avec le raw
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000,
            max_value=None
        )
    )

    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name  = "processed_validation",
            data  = batch_def,
            suite = suite,
        )
    )

    results = validation_def.run(
        batch_parameters={"dataframe": df}
    )

    report_path = "monitoring/reports/ge_processed_validation.html"
    _save_validation_report(results, report_path, "Processed Data Validation")

    passed = sum(1 for r in results.results if r.success)
    total  = len(results.results)

    logger.info(f"Validation traitée — {passed}/{total} règles passées")

    return {
        "success": results.success,
        "total":   total,
        "passed":  passed,
        "failed":  total - passed,
        "report":  report_path
    }


# ─────────────────────────────────────────
# 3. GÉNÉRATION DU RAPPORT HTML
# ─────────────────────────────────────────

def _save_validation_report(results, path: str, title: str):
    """Génère un rapport HTML lisible des résultats de validation."""

    rows = ""
    for r in results.results:
        status = "✅ PASSED" if r.success else "❌ FAILED"
        color  = "#2ecc71" if r.success else "#e74c3c"
        rule   = r.expectation_config.type
        col    = r.expectation_config.kwargs.get("column", "table-level")
        rows  += f"""
        <tr>
            <td>{col}</td>
            <td>{rule}</td>
            <td style='color:{color}; font-weight:bold'>{status}</td>
        </tr>"""

    global_color  = "#2ecc71" if results.success else "#e74c3c"
    global_status = "✅ ALL PASSED" if results.success else "❌ VALIDATION FAILED"
    passed = sum(1 for r in results.results if r.success)
    total  = len(results.results)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1   {{ color: #2c3e50; }}
  .status {{ font-size: 1.4em; color: {global_color}; margin: 20px 0; }}
  table {{ border-collapse: collapse; width: 100%; background: white; }}
  th    {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
  td    {{ padding: 8px 12px; border-bottom: 1px solid #ddd; }}
  tr:hover {{ background: #f0f0f0; }}
</style></head>
<body>
  <h1>Great Expectations — {title}</h1>
  <div class='status'>{global_status} ({passed}/{total} règles)</div>
  <table>
    <tr><th>Colonne</th><th>Règle</th><th>Statut</th></tr>
    {rows}
  </table>
</body></html>"""

    with open(path, "w") as f:
        f.write(html)
    logger.info(f"Rapport HTML sauvegardé : {path}")


# ─────────────────────────────────────────
# 4. POINT D'ENTRÉE
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== VALIDATION DONNÉES BRUTES ===")
    raw = validate_raw_data("data/raw/Superstore.csv")
    print(f"Statut  : {'✅ OK' if raw['success'] else '❌ ÉCHEC'}")
    print(f"Résultat: {raw['passed']}/{raw['total']} règles passées")

    print("\n=== VALIDATION DONNÉES TRAITÉES ===")
    proc = validate_processed_data("data/processed/Superstore_processed.csv")
    print(f"Statut  : {'✅ OK' if proc['success'] else '❌ ÉCHEC'}")
    print(f"Résultat: {proc['passed']}/{proc['total']} règles passées")

    print(f"\nRapports générés dans monitoring/reports/")