from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import pandas as pd
from pydantic import BaseModel, ValidationError

try:
    import great_expectations as gx
except ImportError:  # pragma: no cover - optional dependency
    gx = None


@dataclass
class ValidationReport:
    rows_validated: int
    rows_failed: int
    expectation_success: bool
    errors: list[str]


class DataQualityGate:
    """Combines strict schema validation with lightweight expectation checks."""

    def __init__(self, schema: Type[BaseModel]) -> None:
        self.schema = schema

    def validate(self, frame: pd.DataFrame) -> ValidationReport:
        errors: list[str] = []
        failed = 0
        for idx, row in enumerate(frame.to_dict(orient="records")):
            try:
                self.schema(**row)
            except ValidationError as exc:
                failed += 1
                errors.append(f"row={idx}: {exc.errors()}")

        expectation_success = True
        if gx is not None and not frame.empty:
            expectation_success = self._run_expectations(frame)

        return ValidationReport(
            rows_validated=len(frame),
            rows_failed=failed,
            expectation_success=expectation_success,
            errors=errors,
        )

    def _run_expectations(self, frame: pd.DataFrame) -> bool:
        context = gx.get_context(mode="ephemeral")
        batch = context.data_sources.add_pandas("warehouse").read_dataframe(frame)
        validator = batch.validate_expectation_suite(
            expectation_suite=gx.ExpectationSuite(
                expectations=[
                    gx.expectations.ExpectColumnValuesToNotBeNull(column=frame.columns[0]),
                    gx.expectations.ExpectTableRowCountToBeBetween(
                        min_value=1,
                        max_value=max(len(frame) * 2, 10),
                    ),
                ]
            )
        )
        return bool(validator["success"])

