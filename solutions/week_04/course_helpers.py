"""Course toolbox for the DSAA Machine Learning practicals.

Shared constants and small bookkeeping objects for the notebooks beside this
file. Each block starts in the week that first uses it, so a week's copy holds
what that week and the weeks before it need. It holds no machine learning and
nothing outside the standard library, and it names no path of its own: a block
that reads or writes a file is handed the folder by the notebook.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


# -- Week 3: course plot palette (validated accessible pair) ----------------

PLOT_BLUE = "#1779A8"
PLOT_ORANGE = "#C96A00"


# -- Week 3: cleaning log ----------------------------------------------------


@dataclass(frozen=True)
class CleaningStep:
    """One recorded cleaning decision: what, to which column, why, and how many rows.

    `carries` holds the decision in a form code can apply, anything JSON can
    hold; a step that carries nothing leaves it None.
    """

    column: str
    action: str
    reason: str
    rows_affected: int
    carries: dict | None = None


class CleaningLog:
    """The cleaning decisions made in a notebook, in order.

    The log cleans nothing: the notebook cleans, then records what it did and
    why, and `to_json` writes the record for a later notebook to `load`.
    """

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.steps: list[CleaningStep] = []

    def record(
        self, column: str, action: str, reason: str, rows_affected: int,
        carries: dict | None = None,
    ) -> None:
        """Append one decision; its reason may not be blank."""
        if not reason.strip():
            raise ValueError("every cleaning step needs a stated reason")
        self.steps.append(
            CleaningStep(column, action, reason, int(rows_affected), deepcopy(carries))
        )

    def records(self) -> list[dict]:
        """The steps as plain dictionaries, one per table row, without `carries`."""
        return [
            {
                "column": step.column,
                "action": step.action,
                "reason": step.reason,
                "rows_affected": step.rows_affected,
            }
            for step in self.steps
        ]

    def to_json(self, path) -> None:
        """Write the log, `carries` included, to `path`."""
        payload = {
            "dataset": self.dataset,
            "steps": [
                {
                    "column": step.column,
                    "action": step.action,
                    "reason": step.reason,
                    "rows_affected": step.rows_affected,
                    "carries": step.carries,
                }
                for step in self.steps
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "CleaningLog":
        """Read a log written by `to_json`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        log = cls(payload["dataset"])
        for step in payload["steps"]:
            log.record(step["column"], step["action"], step["reason"],
                       step["rows_affected"], step.get("carries"))
        return log

    def plan(self, column: str):
        """What the step for `column` carries forward, or None if it carries nothing."""
        for step in self.steps:
            if step.column == column:
                return deepcopy(step.carries)
        return None

    def __len__(self) -> int:
        return len(self.steps)
