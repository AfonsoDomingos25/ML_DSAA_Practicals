"""Apply a logged preprocessing recipe inside each training fold.

`load_recipe` returns a frame and the recipe its log configures, and
`FoldPreprocessor` fits that recipe on training rows and transforms any rows.
"""

from __future__ import annotations

import ast
import importlib
import json
import operator
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from category_encoders import CountEncoder, TargetEncoder
from scipy.stats import chi2_contingency, pearsonr, pointbiserialr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


# -------------------------------------------------------------------------
# Reading a recipe and its frame
# -------------------------------------------------------------------------

def _decision(carries, *required):
    """Return the one logged entry holding every required key."""
    matches = [entry for entry in carries
               if all(key in entry for key in required)]
    if len(matches) != 1:
        raise ValueError(f"expected one entry holding {required}, got {len(matches)}")
    return matches[0]

# The arithmetic a logged formula may use.
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def _rebuild(expression, names):
    """Build a logged call or formula from its parse tree, never executing it.

    `names` maps each bare name to a class or a column.
    """
    node = (ast.parse(expression, mode="eval").body
            if isinstance(expression, str) else expression)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.args:
            raise ValueError(f"a logged call is one name with keywords: {ast.unparse(node)}")
        return names(node.func.id)(
            **{keyword.arg: _rebuild(keyword.value, names) for keyword in node.keywords}
        )
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_rebuild(node.left, names),
                                        _rebuild(node.right, names))
    if isinstance(node, ast.Name):
        return names(node.id)
    return ast.literal_eval(node)

def load_recipe(data_path, log_path):
    """Return the frame at `data_path` and the recipe configured by `log_path`.

    Columns the log defines by a formula are added to the frame.
    """
    steps = json.loads(Path(log_path).read_text(encoding="utf-8"))["steps"]
    carries = [step["carries"] for step in steps if step["carries"] is not None]
    dtypes = [entry.get("cast", entry.get("dtypes")) for entry in carries
              if "cast" in entry or "dtypes" in entry]
    if len(dtypes) != 1:
        raise ValueError(f"expected one entry holding the dtypes, got {len(dtypes)}")
    frame = pd.read_csv(data_path, dtype=dtypes[0])
    for entry in carries:
        if "formula" in entry:
            frame[entry["feature"]] = _rebuild(entry["formula"], frame.__getitem__)

    fill = _decision(carries, "fill", "numeric", "categorical")
    transform = _decision(carries, "transform", "columns")
    representation = _decision(carries, "encoding", "scaler")
    if transform["transform"] not in (None, "none", "log1p"):
        raise ValueError(f"unknown transform: {transform['transform']!r}")
    recipe = {
        "numeric": list(representation["numeric"]),
        "categorical": list(representation["categorical"]),
        "dropped": list(representation.get("dropped", [])),
        "fill": dict(fill["fill"]),
        "log1p": (list(transform["columns"])
                  if transform["transform"] == "log1p" else []),
        "outliers": transform["outliers"],
        "encoding": representation["encoding"],
        "drop": representation["drop"],
        "scaler": representation["scaler"],
        "strategy": _decision(carries, "strategy")["strategy"],
    }
    return frame, recipe


# -------------------------------------------------------------------------
# Turning a recipe into an unfitted preprocessor
# -------------------------------------------------------------------------

# Encoders and scalers by the names a recipe uses; each fit clones them.
ENCODERS = {
    "one-hot": OneHotEncoder(handle_unknown="ignore", drop="first",
                             sparse_output=False),
    "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1),
    "count": CountEncoder(normalize=True, handle_unknown=0,
                          handle_missing="value"),
    "target": TargetEncoder(handle_unknown="value", handle_missing="value"),
}
SCALERS = {"none": None, "standard": StandardScaler,
           "min-max": MinMaxScaler, "robust": RobustScaler}

# The training quantiles each outlier treatment clips to; None clips nothing.
OUTLIER_QUANTILES = {"keep": None, "keep everything": None,
                     "winsorize 1/99": (0.01, 0.99)}

class ProportionalImputer:
    """Fill each gap with a label drawn in its column's training proportions."""

    def __init__(self, random_state=None):
        """Store the seed of the draws."""
        self.random_state = random_state

    def fit(self, frame):
        """Learn each column's label proportions."""
        self.shares = {c: frame[c].value_counts(normalize=True) for c in frame}
        return self

    def transform(self, frame):
        """Fill the gaps with labels drawn from the learned proportions."""
        draws = np.random.default_rng(self.random_state)
        frame = frame.copy()
        for column, shares in self.shares.items():
            gaps = frame[column].isna()
            if gaps.any():
                frame.loc[gaps, column] = draws.choice(
                    shares.index, size=gaps.sum(), p=shares.to_numpy())
        return frame

class FoldPreprocessor(TransformerMixin, BaseEstimator):
    """Fit a recipe's preprocessing on training rows and apply it to any rows."""

    def __init__(self, recipe, *, selection="inherit"):
        """Store the recipe; `selection=None` keeps every encoded column."""
        self.recipe = recipe
        self.selection = selection

    def _categorical_block(self, frame, fit=False):
        """Fill the label columns."""
        # scikit-learn imputers read np.nan as missing, but not pd.NA.
        block = frame[self.categorical_].astype(object).fillna(np.nan).infer_objects()
        if self.recipe["fill"]["categorical"] == "own level":
            # The new label and boolean columns need one common type.
            block = block.astype("string").astype(object).fillna(np.nan)
        if fit:
            self.categorical_imputer_.fit(block)
        return pd.DataFrame(self.categorical_imputer_.transform(block),
                            columns=self.categorical_, index=frame.index)

    def _numeric_block(self, frame, fit=False):
        """Clip, fill and log1p the numeric columns."""
        block = frame[self.numeric_].astype(object).fillna(np.nan).infer_objects()
        if fit:
            treatment = self.recipe["outliers"]
            if treatment.startswith("drop"):
                raise ValueError(
                    f"{treatment!r} removes training rows, which a transformer"
                    " cannot do; drop them in the fold loop, where y is available"
                )
            # Quantiles come from the observed values, before filling.
            limits = OUTLIER_QUANTILES[treatment]
            self.outlier_bounds_ = (None if limits is None
                                    else (block.quantile(limits[0]),
                                          block.quantile(limits[1])))
        if self.outlier_bounds_ is not None:
            block = block.clip(*self.outlier_bounds_, axis=1)
        if self.standardiser_ is None:
            if fit:
                self.numeric_imputer_.fit(block)
            filled = self.numeric_imputer_.transform(block)
        else:
            # KNN imputes on standardised columns, then returns to the units.
            if fit:
                self.numeric_imputer_.fit(self.standardiser_.fit(block).transform(block))
            filled = self.standardiser_.inverse_transform(
                self.numeric_imputer_.transform(self.standardiser_.transform(block))
            )
        filled = np.asarray(filled, dtype=float)
        # log1p is undefined below -1, so negatives go to 0 first.
        where = [self.numeric_.index(column) for column in self.recipe["log1p"]]
        filled[:, where] = np.log1p(np.clip(filled[:, where], 0, None))
        return filled

    def fit(self, X, y=None):
        """Learn every step of the recipe from these rows."""
        self.categorical_ = list(self.recipe["categorical"])
        self.numeric_ = list(self.recipe["numeric"])
        missing = set(self.categorical_ + self.numeric_).difference(X.columns)
        if missing:
            raise ValueError(f"recipe columns missing from frame: {sorted(missing)}")

        fill = self.recipe["fill"]
        imputers = {
            "median": SimpleImputer(strategy="median"),
            "mean": SimpleImputer(strategy="mean"),
            "zero": SimpleImputer(strategy="constant", fill_value=0),
            "KNN": KNNImputer(n_neighbors=fill.get("n_neighbors", 5)),
            "iterative": IterativeImputer(max_iter=10,
                                          random_state=fill.get("random_state")),
            "mode": SimpleImputer(strategy="most_frequent"),
            "own level": SimpleImputer(strategy="constant", fill_value="(missing)"),
            "proportional": ProportionalImputer(fill.get("random_state")),
        }
        self.numeric_imputer_ = imputers[fill["numeric"]]
        self.categorical_imputer_ = imputers[fill["categorical"]]
        self.standardiser_ = StandardScaler() if fill.get("standardised") else None

        encoding = self.recipe["encoding"]
        self.encoder_ = clone(ENCODERS[encoding]) if self.categorical_ else None
        if encoding == "one-hot" and self.encoder_ is not None:
            self.encoder_.set_params(drop=self.recipe["drop"])
        if self.encoder_ is not None:
            block = self._categorical_block(X, fit=True)
            self.encoder_.fit(block, y)  # only target encoding reads y
        make = SCALERS[self.recipe["scaler"]]
        self.scaler_ = make() if make is not None and self.numeric_ else None
        numeric_block = self._numeric_block(X, fit=True)
        if self.scaler_ is not None:
            self.scaler_.fit(numeric_block)
        # An encoding other than one-hot is a number, so it is scaled too.
        self.label_scaler_ = None
        if make is not None and self.encoder_ is not None and encoding != "one-hot":
            self.label_scaler_ = make().fit(
                np.asarray(self.encoder_.transform(block), dtype=float)
            )
        labels = (self.encoder_.get_feature_names_out(self.categorical_)
                  if encoding == "one-hot" else self.categorical_)
        self.encoded_names_ = np.asarray([*labels, *self.numeric_], dtype=object)
        self.selection_ = self._fit_selection(X, y)
        self.support_ = (
            np.ones(len(self.encoded_names_), dtype=bool)
            if self.selection_ is None else np.asarray(self.selection_.get_support())
        )
        return self

    def _encoded(self, frame):
        """The encoded, transformed and scaled matrix, before selection."""
        labels = np.asarray(
            self.encoder_.transform(self._categorical_block(frame)), dtype=float
        ) if self.encoder_ is not None else np.empty((len(frame), 0))
        if self.label_scaler_ is not None:
            labels = self.label_scaler_.transform(labels)
        numbers = self._numeric_block(frame)
        if self.scaler_ is not None:
            numbers = self.scaler_.transform(numbers)
        return np.hstack([labels, numbers])

    def _fit_selection(self, X, y):
        """Build the recipe's selection rule and fit it on these rows."""
        spec = (self.recipe["strategy"] if self.selection == "inherit"
                else self.selection)
        if spec is None:
            return None
        if y is None:
            raise ValueError("the selection reads the target: pass y to fit")
        width = len(self.encoded_names_) - len(self.numeric_)
        names = partial(_selector_class,
                        scores=partial(effect_scores, categorical_width=width))
        if spec["kind"] == "vote":
            selector = Vote({name: _rebuild(text, names)
                             for name, text in spec["members"].items()},
                            spec["minimum"])
        elif spec["kind"] == "sequence":
            selector = Sequence(*[(stage["stage"], _rebuild(stage["selector"], names))
                                  for stage in spec["stages"]])
        else:
            raise ValueError(f"unknown strategy kind: {spec['kind']!r}")
        frame = pd.DataFrame(self._encoded(X), columns=self.encoded_names_)
        fitted = selector.fit(frame, y)
        if not np.asarray(fitted.get_support()).any():
            raise ValueError("the selection kept no columns on these rows")
        return fitted

    def transform(self, X):
        """Transform rows with the fitted recipe."""
        check_is_fitted(self, "encoder_")
        return self._encoded(X)[:, self.support_]


# -------------------------------------------------------------------------
# The selection rule the log carries
# -------------------------------------------------------------------------

def effect_scores(matrix, outcome, categorical_width=0):
    """Relevance of each encoded column to the outcome, in [0, 1].

    Against a 0/1 outcome, a 0/1 column among the first `categorical_width`
    takes Cramer's V and any other column |point-biserial r|; against any
    other outcome, every column takes |Pearson r|. A constant column scores 0.
    """
    matrix = np.asarray(matrix, dtype=float)
    if not np.isin(np.asarray(outcome), [0, 1]).all():
        values = np.asarray(outcome, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.abs([pearsonr(matrix[:, j], values).statistic
                             for j in range(matrix.shape[1])])
        return np.nan_to_num(scores)
    target = np.asarray(outcome, dtype=int)
    scores = []
    for position in range(matrix.shape[1]):
        values = matrix[:, position]
        if position < categorical_width and np.isin(values, [0.0, 1.0]).all():
            table = pd.crosstab(pd.Series(values), pd.Series(target))
            if min(table.shape) < 2:
                scores.append(0.0)
                continue
            statistic = chi2_contingency(table, correction=False)[0]
            scores.append(np.sqrt(
                statistic / (table.to_numpy().sum() * (min(table.shape) - 1))))
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                scores.append(abs(pointbiserialr(target, values).statistic))
    return np.nan_to_num(np.asarray(scores, dtype=float))

class CorrelationFilter:
    """Keep up to `k` columns by relevance, skipping any that correlates above
    `redundancy` with a column already kept; None removes either limit."""

    def __init__(self, k, redundancy, scores):
        """`scores(matrix, outcome)` returns each column's relevance."""
        self.k = k
        self.redundancy = redundancy
        self.scores = scores

    def fit(self, frame, outcome):
        """Choose the columns on these rows."""
        matrix = frame.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            relevance = self.scores(matrix, outcome)
            between = np.nan_to_num(np.abs(np.corrcoef(matrix, rowvar=False)))
        keep = []
        for candidate in np.argsort(relevance)[::-1]:
            if len(keep) == self.k:
                break
            if self.redundancy is not None and any(
                between[candidate, chosen] > self.redundancy for chosen in keep
            ):
                continue
            keep.append(int(candidate))
        self.support_ = np.zeros(matrix.shape[1], dtype=bool)
        self.support_[keep] = True
        return self

    def get_support(self):
        """The mask of kept columns."""
        return self.support_

class Sequence:
    """Apply selectors in turn, each to the columns the previous one kept."""

    def __init__(self, *stages):
        """Store the (name, selector) stages in order."""
        self.stages = stages

    def fit(self, frame, outcome):
        """Fit each stage on the columns still kept."""
        support = np.ones(frame.shape[1], dtype=bool)
        for _, stage in self.stages:
            kept = stage.fit(frame.loc[:, support], outcome).get_support()
            support[np.flatnonzero(support)] = kept
        self.support_ = support
        return self

    def get_support(self):
        """The mask of kept columns."""
        return self.support_

class Vote:
    """Keep the columns at least `minimum` of the members select."""

    def __init__(self, members, minimum):
        """Store the named members and the votes a column needs."""
        self.members = members
        self.minimum = minimum

    def fit(self, frame, outcome):
        """Fit every member on these rows and count the votes."""
        votes = sum(np.asarray(member.fit(frame, outcome).get_support(), dtype=int)
                    for member in self.members.values())
        self.support_ = votes >= self.minimum
        return self

    def get_support(self):
        """The mask of kept columns."""
        return self.support_

# Libraries searched for a selector the recipe names.
SELECTION_LIBRARIES = (
    "sklearn.feature_selection",
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.neighbors",
)

def _selector_class(name, scores):
    """Return the selector class a recipe names."""
    if name == "CorrelationFilter":
        return partial(CorrelationFilter, scores=scores)
    for library in SELECTION_LIBRARIES:
        found = getattr(importlib.import_module(library), name, None)
        if found is not None:
            return found
    raise ValueError(f"{name} is not defined here nor in {', '.join(SELECTION_LIBRARIES)}")
