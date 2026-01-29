"""
Data validation utilities for aggregation mode.

Provides pre-upload validation to catch data issues before sending to Monday.com.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import streamlit as st


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "data_type", "missing_value", "duplicate", "outlier"
    message: str
    column: Optional[str] = None
    row_count: int = 0


@dataclass
class ValidationReport:
    """Complete validation report for a DataFrame."""
    is_valid: bool = True
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.error_count += 1
            self.is_valid = False
        elif issue.severity == "warning":
            self.warning_count += 1
        else:
            self.info_count += 1


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> list[ValidationIssue]:
    """
    Check that all required columns are present.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        List of validation issues
    """
    issues = []
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        issues.append(ValidationIssue(
            severity="error",
            category="missing_column",
            message=f"Colonnes requises manquantes: {', '.join(missing)}",
            row_count=len(df),
        ))
    return issues


def validate_null_values(
    df: pd.DataFrame,
    critical_columns: Optional[list[str]] = None,
) -> list[ValidationIssue]:
    """
    Check for null/NaN values in the DataFrame.

    Args:
        df: DataFrame to validate
        critical_columns: Columns where nulls are errors (others are warnings)

    Returns:
        List of validation issues
    """
    issues = []
    critical_columns = critical_columns or ["Conseiller"]

    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            severity = "error" if col in critical_columns else "warning"
            issues.append(ValidationIssue(
                severity=severity,
                category="missing_value",
                message=f"{null_count} valeur(s) vide(s)",
                column=col,
                row_count=null_count,
            ))

    return issues


def validate_duplicate_advisors(
    df: pd.DataFrame,
    advisor_column: str = "Conseiller",
) -> list[ValidationIssue]:
    """
    Check for duplicate advisor names.

    Args:
        df: DataFrame to validate
        advisor_column: Name of the advisor column

    Returns:
        List of validation issues
    """
    issues = []

    if advisor_column not in df.columns:
        return issues

    duplicates = df[df.duplicated(subset=[advisor_column], keep=False)]
    if not duplicates.empty:
        dup_names = duplicates[advisor_column].unique().tolist()
        names_str = ", ".join(str(n) for n in dup_names[:5])
        if len(dup_names) > 5:
            names_str += f", ... (+{len(dup_names) - 5} autres)"

        issues.append(ValidationIssue(
            severity="error",
            category="duplicate",
            message=f"Conseillers en double: {names_str}",
            column=advisor_column,
            row_count=len(duplicates),
        ))

    return issues


def validate_numeric_columns(
    df: pd.DataFrame,
    expected_numeric: Optional[list[str]] = None,
) -> list[ValidationIssue]:
    """
    Check that expected numeric columns contain valid numbers.

    Args:
        df: DataFrame to validate
        expected_numeric: Columns expected to be numeric

    Returns:
        List of validation issues
    """
    issues = []

    # Default numeric columns for aggregation
    if expected_numeric is None:
        expected_numeric = [
            "AE CA", "PA Vendues", "Collected", "Coût", "Dépenses par Conseiller",
            "Leads", "Bonus", "Récompenses", "Total Dépenses", "Profit",
            "CA/Lead", "Profit/Lead", "Ratio Brut", "Ratio Net"
        ]

    for col in expected_numeric:
        if col not in df.columns:
            continue

        # Try to convert to numeric
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        non_numeric_count = numeric_col.isna().sum() - df[col].isna().sum()

        if non_numeric_count > 0:
            issues.append(ValidationIssue(
                severity="warning",
                category="data_type",
                message=f"{non_numeric_count} valeur(s) non numérique(s)",
                column=col,
                row_count=non_numeric_count,
            ))

    return issues


def validate_outliers(
    df: pd.DataFrame,
    numeric_columns: Optional[list[str]] = None,
    z_threshold: float = 3.0,
) -> list[ValidationIssue]:
    """
    Detect statistical outliers in numeric columns.

    Args:
        df: DataFrame to validate
        numeric_columns: Columns to check for outliers
        z_threshold: Z-score threshold for outlier detection

    Returns:
        List of validation issues
    """
    issues = []

    if numeric_columns is None:
        # Auto-detect numeric columns
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().all():
            continue

        mean = values.mean()
        std = values.std()

        if std == 0:
            continue

        z_scores = (values - mean).abs() / std
        outlier_count = (z_scores > z_threshold).sum()

        if outlier_count > 0:
            outlier_values = values[z_scores > z_threshold].head(3).tolist()
            values_str = ", ".join(f"{v:,.2f}" for v in outlier_values)

            issues.append(ValidationIssue(
                severity="info",
                category="outlier",
                message=f"{outlier_count} valeur(s) atypique(s): {values_str}...",
                column=col,
                row_count=outlier_count,
            ))

    return issues


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    critical_columns: Optional[list[str]] = None,
    check_outliers: bool = True,
) -> ValidationReport:
    """
    Perform comprehensive validation on a DataFrame.

    Args:
        df: DataFrame to validate
        required_columns: Columns that must be present
        critical_columns: Columns where nulls are errors
        check_outliers: Whether to check for outliers

    Returns:
        ValidationReport with all issues
    """
    report = ValidationReport(
        row_count=len(df),
        column_count=len(df.columns),
    )

    if df.empty:
        report.add_issue(ValidationIssue(
            severity="error",
            category="empty",
            message="Le DataFrame est vide - aucune donnée à uploader",
        ))
        return report

    # Required columns
    if required_columns:
        for issue in validate_required_columns(df, required_columns):
            report.add_issue(issue)

    # Null values
    for issue in validate_null_values(df, critical_columns or ["Conseiller"]):
        report.add_issue(issue)

    # Duplicate advisors
    for issue in validate_duplicate_advisors(df):
        report.add_issue(issue)

    # Numeric validation
    for issue in validate_numeric_columns(df):
        report.add_issue(issue)

    # Outliers
    if check_outliers:
        for issue in validate_outliers(df):
            report.add_issue(issue)

    return report


def render_validation_report(report: ValidationReport) -> None:
    """
    Render a validation report in Streamlit.

    Args:
        report: ValidationReport to display
    """
    # Summary
    if report.is_valid:
        st.success(f"✅ Validation réussie: {report.row_count} lignes, {report.column_count} colonnes")
    else:
        st.error(f"❌ Validation échouée: {report.error_count} erreur(s) trouvée(s)")

    # Issue counts
    col1, col2, col3 = st.columns(3)
    with col1:
        if report.error_count > 0:
            st.metric("Erreurs", report.error_count, delta=None)
        else:
            st.metric("Erreurs", "0", delta="OK", delta_color="off")
    with col2:
        st.metric("Avertissements", report.warning_count)
    with col3:
        st.metric("Informations", report.info_count)

    # Detailed issues
    if report.issues:
        with st.expander("📋 Détails des problèmes", expanded=report.error_count > 0):
            # Group by severity
            errors = [i for i in report.issues if i.severity == "error"]
            warnings = [i for i in report.issues if i.severity == "warning"]
            infos = [i for i in report.issues if i.severity == "info"]

            if errors:
                st.markdown("**Erreurs (bloquantes):**")
                for issue in errors:
                    col_info = f" (colonne: {issue.column})" if issue.column else ""
                    st.markdown(f"- 🔴 {issue.message}{col_info}")

            if warnings:
                st.markdown("**Avertissements:**")
                for issue in warnings:
                    col_info = f" (colonne: {issue.column})" if issue.column else ""
                    st.markdown(f"- 🟡 {issue.message}{col_info}")

            if infos:
                st.markdown("**Informations:**")
                for issue in infos:
                    col_info = f" (colonne: {issue.column})" if issue.column else ""
                    st.markdown(f"- 🔵 {issue.message}{col_info}")


def validate_and_display(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
) -> bool:
    """
    Validate DataFrame and display results. Returns True if valid.

    Args:
        df: DataFrame to validate
        required_columns: Columns that must be present

    Returns:
        True if validation passed, False otherwise
    """
    report = validate_dataframe(df, required_columns=required_columns)
    render_validation_report(report)
    return report.is_valid
