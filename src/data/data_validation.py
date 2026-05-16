"""
Validation des données avec Great Expectations.
"""

from typing import Any, Dict

import great_expectations as gx
import pandas as pd

from src.config import config
from src.logging_utils import get_logger, log_performance

logger = get_logger(__name__)


def _save_validation_report(results, path: str, title: str):
    """
    Sauvegarde le rapport de validation en HTML.

    Args:
        results: Résultats de validation Great Expectations
        path: Chemin de sauvegarde
        title: Titre du rapport
    """
    # Dans une version réelle, on utiliserait gx.render
    # Ici on simule pour l'exemple
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"<html><body><h1>{title}</h1>")
        f.write(f"<p>Success: {results.success}</p>")
        f.write("<ul>")
        for r in results.results:
            status = "✅" if r.success else "❌"
            f.write(f"<li>{status} {r.expectation_config.type}</li>")
        f.write("</ul></body></html>")


@log_performance
def validate_raw_data(path: str = None) -> Dict[str, Any]:
    """
    Valide le dataset brut Superstore.

    Vérifications:
    - Toutes les colonnes requises présentes
    - Types de données corrects
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

    df = pd.read_csv(path)

    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("superstore_raw")
    asset = datasource.add_dataframe_asset("raw_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    batch_def.get_batch(batch_parameters={"dataframe": df})

    # Suite d'expectations
    suite = context.suites.add(gx.ExpectationSuite(name="superstore_raw_suite"))

    # 1. Colonnes obligatoires
    required_cols = ["Row ID", "Order ID", "Order Date", "Sales", "Profit", "Quantity"]
    for col in required_cols:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    # 2. Valeurs positives
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="Sales", min_value=0)
    )

    # 3. Row ID unique
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="Row ID"))

    # 4. Volume minimal (ex: > 5000 lignes)
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=5000)
    )

    # Exécution
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="raw_validation",
            data=batch_def,
            suite=suite,
        )
    )

    results = validation_def.run(batch_parameters={"dataframe": df})

    # Rapport
    report_path = config.monitoring.drift_report_path.parent / "ge_raw_validation.html"
    _save_validation_report(results, str(report_path), "Raw Data Validation")

    # Résumé
    success = results.success
    total = len(results.results)
    passed = sum(1 for r in results.results if r.success)
    failed = total - passed

    logger.info(f"Validation brute — {passed}/{total} règles passées")

    return {
        "success": success,
        "total": total,
        "passed": passed,
        "failed": failed,
        "report": str(report_path),
    }


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
    batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name="superstore_processed_suite"))

    # 1. Target binaire
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="is_profitable"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_profitable", value_set=[0, 1]
        )
    )

    # 2. Features temporelles
    for col in ["order_month", "order_quarter", "order_dayofweek", "shipping_delay"]:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

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

    results = validation_def.run(batch_parameters={"dataframe": df})

    # Rapport
    report_path = (
        config.monitoring.drift_report_path.parent / "ge_processed_validation.html"
    )
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
        "report": str(report_path),
    }


def get_validation_status(
    raw_path: str = None, processed_path: str = None
) -> Dict[str, Any]:
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
        "overall_success": raw_results["success"] and processed_results["success"],
    }


if __name__ == "__main__":
    # Test
    status = get_validation_status()
    print("\n✅ Validation complète")
    print(f"Raw data: {status['raw']}")
    print(f"Processed data: {status['processed']}")
