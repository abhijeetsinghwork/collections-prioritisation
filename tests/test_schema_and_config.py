"""Schema generation from a synthetic layout workbook, and config validation."""

from __future__ import annotations

import openpyxl
import pydantic
import pytest
import yaml

from src.utils import gen_schema
from src.utils.config import DEFAULT_CONFIG_PATH, Config, load_config
from src.utils.schema import ORIGINATION_FIELDS, PERFORMANCE_FIELDS


def _layout(tmp_path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet"
    ws.append(["TITLE", None, None, None])
    ws.append(["Field Position", "Attribute Name", "Data Type & Format", "Max Length"])
    for r in rows:
        ws.append(r)
    path = tmp_path / "layout.xlsx"
    wb.save(path)
    return path


def test_read_layout_sheet_skips_headers_and_normalises_whitespace(tmp_path):
    path = _layout(tmp_path, [["1", "Loan  Age", "Numeric ", "3"], ["2", "Period", "Date", "6"]])
    fields = gen_schema.read_layout_sheet(path, "Sheet")
    assert [(f.position, f.source_name, f.data_type) for f in fields] == [
        (1, "Loan Age", "Numeric"),
        (2, "Period", "Date"),
    ]


def test_read_layout_sheet_rejects_position_gaps(tmp_path):
    path = _layout(tmp_path, [["1", "A", "Alpha", "1"], ["3", "B", "Alpha", "1"]])
    with pytest.raises(ValueError, match="not contiguous"):
        gen_schema.read_layout_sheet(path, "Sheet")


@pytest.mark.parametrize(
    ("dtype", "length", "column", "kind"),
    [
        ("Numeric", 4, "credit_score", "int"),
        ("Numeric", 12, "orig_upb", "double"),  # long numerics carry decimals in the files
        ("Numeric - 12,2", 12, "current_upb", "double"),
        ("Numeric - 6,3", 6, "orig_interest_rate", "double"),
        ("Date", 6, "first_payment_date", "yyyymm"),
        ("Alpha", 1, "channel", "string"),
        ("Numeric", 2, "zero_balance_code", "string"),  # categorical code, keeps "01"
        ("Numeric", 5, "msa", "string"),
    ],
)
def test_kind_for(dtype, length, column, kind):
    field = gen_schema.LayoutField(1, "x", dtype, length)
    assert gen_schema.kind_for(field, column) == kind


def test_resolve_names_fails_loudly_on_layout_change():
    fields = [gen_schema.LayoutField(1, "Loan Identifier", "Alpha", 12)]
    with pytest.raises(ValueError, match="Brand New Field"):
        gen_schema.resolve_names(
            [*fields, gen_schema.LayoutField(2, "Brand New Field", "Alpha", 1)],
            {"Loan Identifier": "loan_sequence_number"},
            "Sheet",
        )


def test_generated_schema_has_join_key_and_unique_names():
    for fields in (ORIGINATION_FIELDS, PERFORMANCE_FIELDS):
        names = [n for n, _, _ in fields]
        assert "loan_sequence_number" in names
        assert len(names) == len(set(names))


def test_config_loads_and_retained_columns_exist(cfg: Config):
    perf_derived = {"reporting_period", "reporting_year", "dq_months", "dq_bucket", "is_reo"}
    perf_names = {n for n, _, _ in PERFORMANCE_FIELDS} | perf_derived
    orig_names = {n for n, _, _ in ORIGINATION_FIELDS} | {"vintage"}
    assert set(cfg.ingest.retain_performance) <= perf_names
    assert set(cfg.ingest.retain_origination) <= orig_names
    assert set(cfg.ingest.sentinels) <= perf_names | orig_names


def test_config_rejects_unknown_keys(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["spark"]["driver_memroy"] = "4g"  # typo
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(pydantic.ValidationError, match="driver_memroy"):
        load_config(path)
