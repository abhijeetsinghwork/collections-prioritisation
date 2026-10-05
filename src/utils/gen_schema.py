"""Generate src/utils/schema.py from the official SFLLD File Layout workbook.

Field positions and data types come from the workbook, which is the
authoritative source and has been revised across releases. Column names are a
curated mapping keyed on the workbook's attribute names: if Freddie Mac adds,
removes or renames a field, generation fails and names the mismatch instead of
silently shifting every column after it.

Run: python -m src.utils.gen_schema
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import openpyxl

from src.utils.config import load_config

ORIGINATION_SHEET = "Origination Data File"
PERFORMANCE_SHEET = "Monthly Performance Data File"

ORIGINATION_NAMES: dict[str, str] = {
    "Classic FICO®": "credit_score",
    "First Payment Date": "first_payment_date",
    "First Time Homebuyer Indicator": "first_time_homebuyer",
    "Maturity Date": "maturity_date",
    "Metropolitan Statistical Area (MSA) Or Metropolitan Division": "msa",
    "Mortgage Insurance Percentage (MI %)": "mi_pct",
    "Number of Units": "num_units",
    "Occupancy Status": "occupancy_status",
    "Original Combined Loan-to-Value (CLTV)": "orig_cltv",
    "Original Debt-to-Income (DTI) Ratio": "orig_dti",
    "Original UPB": "orig_upb",
    "Original Loan-to-Value (LTV)": "orig_ltv",
    "Original Interest Rate": "orig_interest_rate",
    "Channel": "channel",
    "Prepayment Penalty Indicator": "prepayment_penalty",
    "Amortization Type": "amortization_type",
    "Property State": "property_state",
    "Property Type": "property_type",
    "Postal Code": "postal_code",
    "Loan Identifier": "loan_sequence_number",
    "Loan Purpose": "loan_purpose",
    "Original Loan Term": "orig_loan_term",
    "Number of Borrowers": "num_borrowers",
    "Seller Name": "seller_name",
    "Super Conforming Flag": "super_conforming",
    "Pre-HARP Loan Sequence Number": "pre_harp_loan_sequence_number",
    "Special Eligibility Program": "special_eligibility_program",
    "HARP Indicator": "harp_indicator",
    "Property Valuation Method": "property_valuation_method",
    "Interest Only (I/O) Indicator": "interest_only",
    "VantageScore® 4.0": "vantage_score",
}

PERFORMANCE_NAMES: dict[str, str] = {
    "Loan Identifier": "loan_sequence_number",
    "Period": "monthly_reporting_period",
    "Current Actual UPB": "current_upb",
    "Current Loan Delinquency Status": "delinquency_status",
    "Loan Age": "loan_age",
    "Remaining Months to Legal Maturity": "remaining_months_to_maturity",
    "Underwriting Defect and Major Servicing Defect Settlement Date": "defect_settlement_date",
    "Modification Flag": "modification_flag",
    "Zero Balance Code": "zero_balance_code",
    "Zero Balance Effective Date": "zero_balance_effective_date",
    "Current Interest Rate": "current_interest_rate",
    "Current Non-Interest Bearing UPB": "current_deferred_upb",
    "Due Date of Last Paid Installment (DDLPI)": "ddlpi",
    "MI Recoveries": "mi_recoveries",
    "Net Sales Proceeds": "net_sales_proceeds",
    "Non MI Recoveries": "non_mi_recoveries",
    "Total Expenses": "total_expenses",
    "Legal Costs": "legal_costs",
    "Maintenance and Preservation Costs": "maintenance_costs",
    "Taxes and Insurance": "taxes_and_insurance",
    "Miscellaneous Expenses": "misc_expenses",
    "Actual Loss": "actual_loss",
    "Cumulative Modification Costs": "cumulative_mod_cost",
    "Interest Rate Step Indicator": "step_modification_flag",
    "Payment Deferral Flag": "payment_deferral_flag",
    "Estimated Loan-to-Value (ELTV)": "eltv",
    "Zero Balance Removal UPB": "zero_balance_removal_upb",
    "Delinquent Accrued Interest": "delinquent_accrued_interest",
    "Delinquency Due to Disaster": "delinquency_due_to_disaster",
    "Borrower Assistance Plan": "borrower_assistance_status",
    "Current Period Modification Costs": "current_period_mod_cost",
    "Current Interest Bearing UPB": "current_interest_bearing_upb",
    "Mortgage Insurance Cancellation Indicator": "mi_cancellation_indicator",
    "Servicer Name": "servicer_name",
    "Bankruptcy Cramdown Costs": "bankruptcy_cramdown_costs",
}

# Fields the layout types as "Numeric" that are really categorical codes.
# Casting them to numbers would drop leading zeros ("01") or imply an order.
CATEGORICAL_NUMERIC = {"msa", "zero_balance_code", "property_valuation_method"}

# "Numeric" without an explicit scale is an integer only when it is short;
# longer ones are dollar amounts that the files write with decimals ("0.00").
MAX_INT_LENGTH = 4


@dataclass(frozen=True)
class LayoutField:
    position: int
    source_name: str
    data_type: str
    max_length: int


def read_layout_sheet(path: Path, sheet: str) -> list[LayoutField]:
    """Return the fields listed on one sheet of the layout workbook, in file order."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    fields: list[LayoutField] = []
    for row in ws.iter_rows(values_only=True):
        position, name, dtype, length = (row + (None,) * 4)[:4]
        if position is None or not str(position).strip().isdigit():
            continue  # title and header rows
        fields.append(
            LayoutField(
                position=int(str(position).strip()),
                source_name=" ".join(str(name).split()),
                data_type=" ".join(str(dtype).split()),
                max_length=int(str(length).strip()),
            )
        )
    positions = [f.position for f in fields]
    if positions != list(range(1, len(fields) + 1)):
        raise ValueError(f"{sheet}: field positions are not contiguous from 1: {positions}")
    return fields


def kind_for(field: LayoutField, column: str) -> str:
    """Return the ingest kind for a field: string, int, double or yyyymm."""
    dtype = field.data_type.lower()
    if column in CATEGORICAL_NUMERIC:
        return "string"
    if dtype == "date":
        return "yyyymm"
    if dtype.startswith("numeric"):
        if re.search(r"\d+\s*,\s*\d+", dtype):
            return "double"
        return "int" if field.max_length <= MAX_INT_LENGTH else "double"
    return "string"


def resolve_names(fields: list[LayoutField], names: dict[str, str], sheet: str) -> list[str]:
    """Map layout attribute names to column names, failing on any mismatch."""
    layout_names = [f.source_name for f in fields]
    unknown = [n for n in layout_names if n not in names]
    missing = [n for n in names if n not in layout_names]
    if unknown or missing:
        raise ValueError(
            f"{sheet}: layout and name map disagree.\n"
            f"  in layout but not mapped: {unknown}\n"
            f"  mapped but not in layout: {missing}"
        )
    return [names[n] for n in layout_names]


def render_module(layout_path: Path, sheets: dict[str, list[tuple[str, str, str]]]) -> str:
    """Return the source text of the generated schema module."""
    lines = [
        '"""SFLLD file schemas. GENERATED by src/utils/gen_schema.py - do not edit by hand.',
        "",
        f"Source: {layout_path.name}",
        '"""',
        "",
        "# (column name, ingest kind, layout attribute name), in file order.",
        "# Kinds: string | int | double | yyyymm (parsed to first-of-month date).",
        "",
    ]
    for const, rows in sheets.items():
        lines.append(f"{const}: list[tuple[str, str, str]] = [")
        lines.extend(
            f"    ({json.dumps(col)}, {json.dumps(kind)}, {json.dumps(src, ensure_ascii=False)}),"
            for col, kind, src in rows
        )
        lines.append("]")
        lines.append("")
    return "\n".join(lines)


def build_schema_source(layout_path: Path) -> str:
    """Parse the layout workbook and return the generated module source."""
    sheets: dict[str, list[tuple[str, str, str]]] = {}
    for const, sheet, names in (
        ("ORIGINATION_FIELDS", ORIGINATION_SHEET, ORIGINATION_NAMES),
        ("PERFORMANCE_FIELDS", PERFORMANCE_SHEET, PERFORMANCE_NAMES),
    ):
        fields = read_layout_sheet(layout_path, sheet)
        columns = resolve_names(fields, names, sheet)
        sheets[const] = [
            (col, kind_for(f, col), f.source_name) for f, col in zip(fields, columns, strict=True)
        ]
    return render_module(layout_path, sheets)


def main() -> int:
    cfg = load_config()
    source = build_schema_source(cfg.paths.layout_file)
    cfg.paths.schema_module.write_text(source)
    print(f"wrote {cfg.paths.schema_module} from {cfg.paths.layout_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
