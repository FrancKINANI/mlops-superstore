# src/data/data_validation.py
"""
Validation des données avec Great Expectations.
Vérifie que les données respectent le contrat attendu
avant d'entrer dans le pipeline de preprocessing.
"""

import great_expectations as gx
import pandas as pd
import numpy as np
import os, sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 1. VALIDATION DES DONNÉES BRUTES
# ─────────────────────────────────────────

def validate_raw_data(path: str) -> dict:
    """
    Valide le dataset brut Superstore.
    Retourne un dict avec le statut et les résultats.
    """
    logger.info(f"Validation des données brutes : {path}")
    df = pd.read_csv(path, encoding='latin-1')

    # Crée le contexte GX en mémoire (pas besoin de fichiers de config)
    context = gx.get_context(mode="ephemeral")

    # Source de données
    datasource = context.data_sources.add_pandas("superstore_raw")
    asset      = datasource.add_dataframe_asset("raw_asset")
    batch_def  = asset.add_batch_definition_whole_dataframe("batch")
    batch      = batch_def.get_batch(batch_parameters={"dataframe": df})

    # ── Suite d'expectations ──
    suite = context.suites.add(
        gx.ExpectationSuite(name="superstore_raw_suite")
    )

    # 1. Colonnes obligatoires présentes
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

    # 5. Category dans les valeurs attendues
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Category",
            value_set=["Furniture", "Office Supplies", "Technology"]
        )
    )

    # 6. Region dans les valeurs attendues
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Region",
            value_set=["West", "East", "Central", "South"]
        )
    )

    # 7. Segment dans les valeurs attendues
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Segment",
            value_set=["Consumer", "Corporate", "Home Office"]
        )
    )

    # 8. Pas de nulls sur les colonnes critiques
    for col in ["Sales", "Profit", "Discount", "Category", "Region"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # 9. Volume minimum de données
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=1000,
            max_value=None
        )
    )

    # 10. Unicité de Row ID
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="Row ID")
    )

    # ── Exécution de la validation ──
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name       = "raw_validation",
            data       = batch_def,
            suite      = suite,
        )
    )

    results = validation_def.run(
        batch_parameters={"dataframe": df}
    )

    # ── Rapport HTML ──
    os.makedirs("monitoring/reports", exist_ok=True)
    report_path = "monitoring/reports/ge_raw_validation.html"

    _save_validation_report(results, report_path, "Raw Data Validation")

    # ── Résumé ──
    success        = results.success
    total          = len(results.results)
    passed         = sum(1 for r in results.results if r.success)
    failed         = total - passed

    logger.info(f"Validation brute — {passed}/{total} règles passées")
    if not success:
        failed_rules = [
            r.expectation_config.type
            for r in results.results if not r.success
        ]
        logger.warning(f"Règles échouées : {failed_rules}")

    return {
        "success": success,
        "total":   total,
        "passed":  passed,
        "failed":  failed,
        "report":  report_path
    }


# ─────────────────────────────────────────
# 2. VALIDATION DES DONNÉES TRAITÉES
# ─────────────────────────────────────────

def validate_processed_data(path: str) -> dict:
    """
    Valide le dataset après preprocessing.
    Vérifie que les features engineerées sont correctes.
    """
    logger.info(f"Validation des données traitées : {path}")
    df = pd.read_csv(path)

    context   = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("superstore_processed")
    asset      = datasource.add_dataframe_asset("processed_asset")
    batch_def  = asset.add_batch_definition_whole_dataframe("batch")
    batch      = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(
        gx.ExpectationSuite(name="superstore_processed_suite")
    )

    # 1. Variable cible présente et binaire
    suite.add_expectation(
        gx.expectations.ExpectColumnToExist(column="is_profitable")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_profitable", value_set=[0, 1]
        )
    )

    # 2. Features temporelles présentes
    for col in ["order_month", "order_quarter", "order_dayofweek", "shipping_delay"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

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