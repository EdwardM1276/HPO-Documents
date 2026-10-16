# -*- coding: utf-8 -*-
"""Finance-fraud HPO benchmark and analysis.

Default mode analyses the recorded workbook without fitting models.
Use --mode benchmark only to run a new experiment; outputs are saved separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    HalvingGridSearchCV,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from optuna.distributions import CategoricalDistribution
    try:
        from optuna_integration import OptunaSearchCV
    except ImportError:
        from optuna.integration import OptunaSearchCV
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Categorical

    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False

try:
    from pyswarm import pso

    SWARM_AVAILABLE = True
except ImportError:
    SWARM_AVAILABLE = False

warnings.filterwarnings("ignore")


CONFIG = {
    "protocol_id": "foldlocal_equal100_v2",
    "random_state": 42,
    "outer_splits": 10,
    "test_size": 0.20,
    "inner_cv": 3,
    "n_trials": 100,
    "pso_swarm_size": 20,
    "pso_iterations": 4,
    "halving_candidates": 66,
    "halving_factor": 3,
    "halving_min_res": 20,
    "halving_max_res": 1620,
    "ga_population": 10,
    "ga_generations": 9,
    "ga_tournament": 3,
    "ga_crossover": 0.8,
    "ga_mutation": 0.1,
    "output_dir": "finance_hpo_new_runs",
    "results_file": "finance_benchmark_results.csv",
    "checkpoint_file": "finance_benchmark_checkpoint.csv",
}

ALGORITHMS = ["LR", "NB", "kNN", "DT", "RF", "MLP"]
TUNERS = ["RandomSearch", "HalvingSearch", "Bayesian", "Evolutionary", "Swarm"]

RESULTS_PATH = os.path.join(CONFIG["output_dir"], CONFIG["results_file"])
CHECKPOINT_PATH = os.path.join(CONFIG["output_dir"], CONFIG["checkpoint_file"])


def cast_bool_to_int(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("int8")
    return df


class Choice:
    """Finite categorical sampler that samples with replacement."""

    def __init__(self, values):
        self.values = tuple(values)

    def rvs(self, random_state=None):
        if hasattr(random_state, "randint"):
            index = random_state.randint(len(self.values))
        else:
            index = np.random.default_rng(random_state).integers(len(self.values))
        return self.values[index]


def with_replacement(space):
    return {name: Choice(values) for name, values in space.items()}


def build_pipeline(model):
    numeric_cols = [
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
    ]
    binary_cols = [
        "repeat_retailer",
        "used_chip",
        "used_pin_number",
        "online_order",
    ]
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_cols),
            ("binary", "passthrough", binary_cols),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", model)])


class ConfiguredLogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        C=1.0,
        solver_penalty=("l2", "lbfgs"),
        max_iter=1000,
        class_weight=None,
        random_state=42,
    ):
        self.C = C
        self.solver_penalty = solver_penalty
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        penalty, solver = self.solver_penalty.split("|")
        params = {
            "C": self.C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
        }
        if penalty == "elasticnet":
            params["l1_ratio"] = 0.5
        self.estimator_ = LogisticRegression(**params).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class ConfiguredGaussianNB(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        priors = None
        if not self.fit_prior:
            priors = np.full(len(np.unique(y)), 1 / len(np.unique(y)))
        self.estimator_ = GaussianNB(
            var_smoothing=self.alpha,
            priors=priors,
        ).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class ConfiguredMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layer_code="50",
        activation="relu",
        learning_rate_init=0.001,
        alpha=0.0001,
        max_iter=500,
        random_state=42,
    ):
        self.hidden_layer_code = hidden_layer_code
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        hidden_layer_sizes = tuple(
            int(width) for width in self.hidden_layer_code.split(",")
        )
        self.estimator_ = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def load_finance():
    """Finance only: OpenML ID 45955."""
    import openml

    print("[Finance] Loading OpenML ID 45955...")
    dataset = openml.datasets.get_dataset(45955, download_data=True)
    frame, _, _, _ = dataset.get_data(dataset_format="dataframe")

    X = frame.drop(columns=["fraud"])
    y = frame["fraud"].astype(int).values

    X = cast_bool_to_int(X)
    binary_cols = ["repeat_retailer", "used_chip", "used_pin_number", "online_order"]
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].astype("int8")

    X = cast_bool_to_int(X)
    print(
        f"[Finance] {X.shape[0]:,} samples, {X.shape[1]} features. "
        f"Fraud rate: {y.mean():.4%}"
    )
    return X, y


def get_model_and_search_space(algo_name, random_state=42):
    if algo_name == "LR":
        model = ConfiguredLogisticRegression(random_state=random_state)
        discrete = {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver_penalty": [
                "l1|liblinear",
                "l1|saga",
                "l2|liblinear",
                "l2|lbfgs",
                "l2|saga",
                "elasticnet|saga",
            ],
            "max_iter": [100, 500, 1000],
            "class_weight": ["balanced", None],
        }
        continuous = {key: list(values) for key, values in discrete.items()}
    elif algo_name == "NB":
        model = ConfiguredGaussianNB()
        discrete = {
            "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
            "fit_prior": [True, False],
        }
        continuous = {key: list(values) for key, values in discrete.items()}
    elif algo_name == "kNN":
        model = KNeighborsClassifier()
        discrete = {
            "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "weights": ["uniform", "distance"],
        }
        continuous = {key: list(values) for key, values in discrete.items()}
    elif algo_name == "DT":
        model = DecisionTreeClassifier(random_state=random_state)
        discrete = {
            "max_depth": [5, 10, 15, 20, 30, 50, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
            "class_weight": ["balanced", None],
        }
        continuous = {key: list(values) for key, values in discrete.items()}
    elif algo_name == "RF":
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        discrete = {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [5, 10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", None],
        }
        continuous = {key: list(values) for key, values in discrete.items()}
    elif algo_name == "MLP":
        model = ConfiguredMLPClassifier(random_state=random_state)
        discrete = {
            "hidden_layer_code": ["50", "100", "50,50", "100,50"],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 1e-2, 1e-1],
            "activation": ["relu", "tanh", "logistic"],
            "max_iter": [200, 500, 1000],
        }
        continuous = {key: list(values) for key, values in discrete.items()}
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    return model, discrete, continuous


def run_random_search(model, space, X_tr, y_tr, inner_cv, n_trials, seed):
    tuner = RandomizedSearchCV(
        estimator=clone(model),
        param_distributions=with_replacement(space),
        n_iter=n_trials,
        cv=inner_cv,
        scoring="f1_macro",
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    tuner.fit(X_tr, y_tr)
    if len(tuner.cv_results_["params"]) != n_trials:
        raise RuntimeError("Random search did not use the configured trial budget.")
    return tuner.best_estimator_, tuner.best_params_, time.time() - t0


def run_halving_search(model, space, X_tr, y_tr, inner_cv, seed):
    initial_candidates = list(
        ParameterSampler(
            with_replacement(space),
            n_iter=CONFIG["halving_candidates"],
            random_state=seed,
        )
    )
    candidate_grid = [
        {name: [value] for name, value in candidate.items()}
        for candidate in initial_candidates
    ]
    tuner = HalvingGridSearchCV(
        estimator=clone(model),
        param_grid=candidate_grid,
        factor=CONFIG["halving_factor"],
        min_resources=CONFIG["halving_min_res"],
        max_resources=CONFIG["halving_max_res"],
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    tuner.fit(X_tr, y_tr)
    if len(tuner.cv_results_["params"]) != CONFIG["n_trials"]:
        raise RuntimeError(
            "Halving search did not use the configured 100 candidate evaluations."
        )
    return tuner.best_estimator_, tuner.best_params_, time.time() - t0


def run_bayesian_search(model, continuous_space, X_tr, y_tr, inner_cv, n_trials, seed):
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required to run the Bayesian search.")

    optuna_space = {}
    for key, value in continuous_space.items():
        optuna_space[key] = CategoricalDistribution(value)

    tuner = OptunaSearchCV(
        estimator=clone(model),
        param_distributions=optuna_space,
        cv=inner_cv,
        scoring="f1_macro",
        n_trials=n_trials,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    tuner.fit(X_tr, y_tr)
    if len(tuner.cv_results_["params"]) != n_trials:
        raise RuntimeError("Bayesian search did not use the configured trial budget.")
    return tuner.best_estimator_, tuner.best_params_, time.time() - t0


def run_genetic_search(model, continuous_space, X_tr, y_tr, inner_cv, seed):
    if not GENETIC_AVAILABLE:
        raise ImportError("sklearn-genetic-opt is required to run evolutionary search.")

    param_grid = {}
    for key, value in continuous_space.items():
        param_grid[key] = Categorical(value)

    tuner = GASearchCV(
        estimator=clone(model),
        cv=inner_cv,
        scoring="f1_macro",
        param_grid=param_grid,
        population_size=CONFIG["ga_population"],
        generations=CONFIG["ga_generations"],
        tournament_size=CONFIG["ga_tournament"],
        crossover_probability=CONFIG["ga_crossover"],
        mutation_probability=CONFIG["ga_mutation"],
        elitism=True,
        random_state=seed,
        n_jobs=-1,
        verbose=False,
    )
    t0 = time.time()
    tuner.fit(X_tr, y_tr)
    if len(tuner.cv_results_["params"]) != CONFIG["n_trials"]:
        raise RuntimeError(
            "Evolutionary search did not use the configured 100 candidate evaluations."
        )
    return tuner.best_estimator_, tuner.best_params_, time.time() - t0


def run_swarm_search(model, continuous_space, X_tr, y_tr, inner_cv, seed):
    if not SWARM_AVAILABLE:
        raise ImportError("pyswarm is required to run particle swarm optimization.")

    np.random.seed(seed)
    objective_calls = 0
    param_names, bounds, categories = [], [], {}
    for key, value in continuous_space.items():
        param_names.append(key)
        bounds.append((0, len(value) - 1))
        categories[key] = value

    def decode(vector):
        params = {}
        for i, name in enumerate(param_names):
            if name in categories:
                idx = int(round(vector[i]))
                idx = max(0, min(idx, len(categories[name]) - 1))
                params[name] = categories[name][idx]
        return params

    def objective(vector):
        nonlocal objective_calls
        objective_calls += 1
        params = decode(vector)
        candidate = clone(model)
        candidate.set_params(**params)
        scores = cross_val_score(
            candidate, X_tr, y_tr, cv=inner_cv, scoring="f1_macro", n_jobs=-1
        )
        return -np.mean(scores)

    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    t0 = time.time()
    best_vector, _ = pso(
        objective,
        lb=lb,
        ub=ub,
        swarmsize=CONFIG["pso_swarm_size"],
        maxiter=CONFIG["pso_iterations"],
        minfunc=-1.0,
        minstep=-1.0,
        debug=False,
    )
    if objective_calls != CONFIG["n_trials"]:
        raise RuntimeError(
            f"PSO used {objective_calls} evaluations; expected {CONFIG['n_trials']}."
        )
    best_params = decode(best_vector)
    best_model = clone(model).set_params(**best_params).fit(X_tr, y_tr)
    return best_model, best_params, time.time() - t0


def run_tuner(name, model, discrete_space, continuous_space, X_tr, y_tr, inner_cv, n_trials, seed):
    if name == "RandomSearch":
        return run_random_search(model, discrete_space, X_tr, y_tr, inner_cv, n_trials, seed)
    if name == "HalvingSearch":
        return run_halving_search(model, discrete_space, X_tr, y_tr, inner_cv, seed)
    if name == "Bayesian":
        return run_bayesian_search(model, continuous_space, X_tr, y_tr, inner_cv, n_trials, seed)
    if name == "Evolutionary":
        return run_genetic_search(model, continuous_space, X_tr, y_tr, inner_cv, seed)
    if name == "Swarm":
        return run_swarm_search(model, continuous_space, X_tr, y_tr, inner_cv, seed)
    raise ValueError(name)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_csv(CHECKPOINT_PATH)
        if "protocol_id" not in df.columns:
            return [], set()
        df = df[df["protocol_id"] == CONFIG["protocol_id"]].copy()
        df["key"] = df["algorithm"] + "_" + df["tuner"] + "_" + df["split"].astype(str)
        return df.to_dict("records"), set(df["key"].unique())
    return [], set()


def save_checkpoint(results):
    if results:
        pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)


def evaluate_model(trained_model, X_te, y_te):
    y_pred = trained_model.predict(X_te)
    metrics = {
        "f1_macro": f1_score(y_te, y_pred, average="macro"),
        "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
    }
    try:
        proba = trained_model.predict_proba(X_te)
        metrics["roc_auc"] = roc_auc_score(y_te, proba[:, 1])
    except Exception:
        metrics["roc_auc"] = np.nan
    return metrics


def run_benchmark(X, y):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    results, completed = load_checkpoint()
    cv = StratifiedShuffleSplit(
        n_splits=CONFIG["outer_splits"],
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
    )

    print("FINANCE-ONLY HPO BENCHMARK")
    print(f"  Algorithms : {ALGORITHMS}")
    print(f"  Tuners     : {TUNERS}")
    print(f"  Splits     : {CONFIG['outer_splits']}")
    print(f"  Trials     : {CONFIG['n_trials']}")

    for split_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        split_id = split_idx + 1
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        print(f"\n--- Split {split_id}/{CONFIG['outer_splits']} ---")
        for algo in ALGORITHMS:
            for tuner in TUNERS:
                key = f"{algo}_{tuner}_{split_id}"
                if key in completed:
                    print(f"    {algo:4s} | {tuner:14s} | skip")
                    continue
                model, disc_space, cont_space = get_model_and_search_space(
                    algo, random_state=CONFIG["random_state"]
                )
                model = build_pipeline(model)
                disc_space = {
                    f"classifier__{name}": values
                    for name, values in disc_space.items()
                }
                cont_space = {
                    f"classifier__{name}": values
                    for name, values in cont_space.items()
                }
                seed = CONFIG["random_state"] + split_idx
                trained, best_params, runtime = run_tuner(
                    tuner,
                    model,
                    disc_space,
                    cont_space,
                    X_tr,
                    y_tr,
                    CONFIG["inner_cv"],
                    CONFIG["n_trials"],
                    seed,
                )
                metrics = evaluate_model(trained, X_te, y_te)
                results.append(
                    {
                        "protocol_id": CONFIG["protocol_id"],
                        "domain": "finance",
                        "algorithm": algo,
                        "tuner": tuner,
                        "split": split_id,
                        "runtime_sec": float(runtime),
                        "candidate_evaluations": CONFIG["n_trials"],
                        "inner_cv_fits": CONFIG["n_trials"] * CONFIG["inner_cv"],
                        "best_params": str(best_params),
                        **metrics,
                    }
                )
                save_checkpoint(results)
                print(
                    f"    {algo:4s} | {tuner:14s} | "
                    f"F1={metrics['f1_macro']:.5f} t={runtime:.2f}s"
                )

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved: {RESULTS_PATH}")
    return df




# ----------------------------------------------------------------------
# Analysis of the recorded workbook
# ----------------------------------------------------------------------

TUNER_ORDER = ["Random Search", "Halving Search", "Bayesian", "Evolutionary", "Particle Swarm"]
TUNER_NAMES = {
    "randomsearch": "Random Search", "random search": "Random Search", "random": "Random Search",
    "halvingsearch": "Halving Search", "halving search": "Halving Search",
    "successive halving": "Halving Search", "bayesian": "Bayesian", "tpe": "Bayesian",
    "evolutionary": "Evolutionary", "genetic": "Evolutionary",
    "swarm": "Particle Swarm", "particleswarm": "Particle Swarm", "particle swarm": "Particle Swarm", "pso": "Particle Swarm",
}


def read_results(path, sheet=0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Workbook not found: {path}")
    if path.lower().endswith('.csv'):
        data = pd.read_csv(path)
    else:
        sheet = int(sheet) if str(sheet).isdigit() else sheet
        data = pd.read_excel(path, sheet_name=sheet)

    data.columns = [str(c).strip().lower().replace('-', '_').replace(' ', '_') for c in data.columns]
    data = data.rename(columns={"f1macro": "f1_macro", "runtime": "runtime_sec", "split_id": "split", "outer_split": "split"})
    required = {"algorithm", "tuner", "split", "f1_macro", "runtime_sec"}
    if not required.issubset(data.columns):
        raise ValueError(f"Workbook must contain {sorted(required)}; found {list(data.columns)}")
    data["algorithm"] = data["algorithm"].astype(str).str.strip()
    data["tuner"] = data["tuner"].astype(str).str.strip().str.lower().map(TUNER_NAMES)
    if data["tuner"].isna().any():
        raise ValueError("Unrecognised tuner name(s): " + str(data.loc[data.tuner.isna(), "tuner"].tolist()))
    for col in ["split", "f1_macro", "runtime_sec"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
    data["split"] = data["split"].astype(int)
    if not np.isfinite(data[["f1_macro", "runtime_sec"]].to_numpy()).all() or (data.runtime_sec <= 0).any():
        raise ValueError("F1 values must be finite and runtimes must be finite and positive.")

    expected = {(a, t, i) for a in ALGORITHMS for t in TUNER_ORDER for i in range(1, 11)}
    found = set(zip(data.algorithm, data.tuner, data.split))
    if len(data) != 300 or found != expected:
        raise ValueError(f"Expected 300 unique runs (6 algorithms x 5 tuners x 10 splits); got {len(data)} rows.")
    return data.sort_values(["algorithm", "tuner", "split"],
        key=lambda col: col.map({v: i for i, v in enumerate(ALGORITHMS)} if col.name == "algorithm" else
                                {v: i for i, v in enumerate(TUNER_ORDER)} if col.name == "tuner" else {})
        if col.name in ["algorithm", "tuner"] else col).reset_index(drop=True)


def kendall_w(matrix):
    matrix = np.asarray(matrix, dtype=float)
    n, judges = matrix.shape
    ranks = np.column_stack([stats.rankdata(matrix[:, j], method="average") for j in range(judges)])
    sums = ranks.sum(axis=1)
    w = 12 * ((sums - judges * (n + 1) / 2) ** 2).sum() / (judges ** 2 * (n ** 3 - n))
    return w, judges * (n - 1) * w, n - 1


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def analyse_results(data, outdir, include_anova=False):
    outdir = os.fspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    combo = data.groupby(["algorithm", "tuner"], sort=False).agg(
        n=("f1_macro", "size"), mean_f1_macro=("f1_macro", "mean"), sd_f1_macro=("f1_macro", "std"),
        mean_runtime_sec=("runtime_sec", "mean"), sd_runtime_sec=("runtime_sec", "std"),
        median_runtime_sec=("runtime_sec", "median"), min_runtime_sec=("runtime_sec", "min"),
        max_runtime_sec=("runtime_sec", "max")).reset_index()

    algo_rows = []
    for alg in ALGORITHMS:
        d = data[data.algorithm == alg]
        means = combo.loc[combo.algorithm == alg, "mean_f1_macro"]
        algo_rows.append({"algorithm": alg, "n_recorded_scores": len(d),
            "mean_f1_macro_all_tuners_and_splits": d.f1_macro.mean(), "sd_f1_macro_sample": d.f1_macro.std(ddof=1),
            "max_tuner_spread": means.max() - means.min()})
    by_algo = pd.DataFrame(algo_rows)

    f0, f1 = combo.mean_f1_macro.min(), combo.mean_f1_macro.max()
    t0, t1 = combo.mean_runtime_sec.min(), combo.mean_runtime_sec.max()
    ranked = combo.copy()
    ranked["normalised_f1"] = (ranked.mean_f1_macro - f0) / (f1 - f0) if f1 != f0 else .5
    ranked["normalised_runtime_preference"] = (t1 - ranked.mean_runtime_sec) / (t1 - t0) if t1 != t0 else .5
    ranked["ranked_balance"] = .5 * (ranked.normalised_f1 + ranked.normalised_runtime_preference)
    ranked = ranked.sort_values(["ranked_balance", "mean_f1_macro", "mean_runtime_sec"],
                                ascending=[False, False, True]).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    vals = combo[["mean_f1_macro", "mean_runtime_sec"]].to_numpy()
    keep = []
    for i, (f, t) in enumerate(vals):
        if not any(j != i and fj >= f and tj <= t and (fj > f or tj < t) for j, (fj, tj) in enumerate(vals)):
            keep.append(i)
    pareto = combo.iloc[keep].sort_values(["mean_runtime_sec", "mean_f1_macro"], ascending=[True, False]).copy()
    pareto.insert(0, "pareto_order_by_runtime", np.arange(1, len(pareto) + 1))

    runtime = np.array([[combo.loc[(combo.algorithm == a) & (combo.tuner == t), "mean_runtime_sec"].iloc[0]
                         for t in TUNER_ORDER] for a in ALGORITHMS])
    concordance = []
    for name, matrix in [("Tuner rankings across classifier families", runtime.T),
                         ("Classifier rankings across tuning methods", runtime)]:
        w, chi, df = kendall_w(matrix)
        concordance.append({"analysis": name, "kendall_W": w, "chi_square_approx": chi, "df": df,
                            "p_approx": stats.chi2.sf(chi, df)})

    # Split-paired Friedman and Nemenyi analyses; interpret as exploratory because splits overlap.
    k, n = len(TUNER_ORDER), CONFIG["outer_splits"]
    q = stats.studentized_range.ppf(.95, k, np.inf)
    cd = q / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n))
    friedman, avg_ranks, nemenyi = [], [], []
    matrices = {}
    for alg in ALGORITHMS:
        mat = data[data.algorithm == alg].pivot(index="split", columns="tuner", values="f1_macro").reindex(
            index=range(1, n + 1), columns=TUNER_ORDER).to_numpy()
        matrices[alg] = mat
        ranks = np.vstack([stats.rankdata(-row, method="average") for row in mat])
        for tuner, rank in zip(TUNER_ORDER, ranks.mean(axis=0)):
            avg_ranks.append({"algorithm": alg, "tuner": tuner, "average_rank_lower_is_better": rank})
        exact = all(np.array_equal(mat[:, 0], mat[:, j]) for j in range(1, k))
        test = stats.friedmanchisquare(*(mat[:, j] for j in range(k))) if not exact else None
        raw_p = test.pvalue if test else np.nan
        friedman.append({"algorithm": alg, "n_matched_splits": n,
            "friedman_chi_square": test.statistic if test else np.nan, "df": k - 1, "p_raw": raw_p,
            "p_bonferroni_across_6_families": min(raw_p * len(ALGORITHMS), 1) if np.isfinite(raw_p) else np.nan,
            "nemenyi_critical_difference": cd,
            "note": "Exploratory; overlapping splits." if not exact else "All five split-level score sequences are identical; rank variance is zero."})
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(ranks[:, i].mean() - ranks[:, j].mean())
                nemenyi.append({"algorithm": alg, "tuner_1": TUNER_ORDER[i], "tuner_2": TUNER_ORDER[j],
                    "absolute_average_rank_difference": diff, "critical_difference_0_05": cd,
                    "exceeds_nemenyi_cd": bool(diff > cd),
                    "interpretation": "Exploratory only; overlapping resamples limit population inference."})

    # Corrected-resampling paired TOST against Halving Search (delta = 0.0005 F1).
    tests, ties, zero_var = [], [], []
    delta = .0005
    correction = 1 / n + CONFIG["test_size"] / (1 - CONFIG["test_size"])
    tcrit = stats.t.ppf(.95, n - 1)
    for alg, mat in matrices.items():
        halving = mat[:, TUNER_ORDER.index("Halving Search")]
        for tuner in ["Random Search", "Bayesian", "Evolutionary", "Particle Swarm"]:
            diff = mat[:, TUNER_ORDER.index(tuner)] - halving
            mean, var = diff.mean(), diff.var(ddof=1)
            if np.all(diff == 0):
                ties.append({"algorithm": alg, "tuner": tuner, "reference_tuner": "Halving Search", "n_splits": n,
                    "status": "Exact observed tie; no inferential p-value assigned.", "mean_difference_tuner_minus_halving": 0.0})
                continue
            if var <= 0:
                zero_var.append({"algorithm": alg, "tuner": tuner, "reference_tuner": "Halving Search", "n_splits": n,
                    "mean_difference_tuner_minus_halving": mean,
                    "status": "Non-identical constant difference; corrected SE not estimable."})
                continue
            se = math.sqrt(correction * var)
            tl, tu = (mean + delta) / se, (mean - delta) / se
            pl, pu = stats.t.sf(tl, n - 1), stats.t.cdf(tu, n - 1)
            tests.append({"algorithm": alg, "tuner": tuner, "reference_tuner": "Halving Search", "n_splits": n,
                "equivalence_margin_abs": delta, "mean_difference_tuner_minus_halving": mean,
                "sample_sd_of_paired_differences": math.sqrt(var), "corrected_standard_error": se,
                "corrected_resampling_factor": correction, "t_lower": tl, "p_lower": pl, "t_upper": tu,
                "p_upper": pu, "p_tost_raw": max(pl, pu), "ci90_corrected_low_pointwise": mean - tcrit * se,
                "ci90_corrected_high_pointwise": mean + tcrit * se})
    if tests:
        for row, p_adj in zip(tests, holm([r["p_tost_raw"] for r in tests])):
            row["p_tost_holm_adjusted"] = p_adj
            row["equivalent_after_holm_alpha_0_05"] = bool(p_adj < .05)
    tost = pd.DataFrame(tests)
    eq_summary = (tost.groupby("algorithm", sort=False).agg(n_nonidentical_tests=("tuner", "count"),
                   n_equivalent_after_holm=("equivalent_after_holm_alpha_0_05", "sum")).reset_index()
                  if not tost.empty else pd.DataFrame(columns=["algorithm", "n_nonidentical_tests", "n_equivalent_after_holm"]))
    tie_df = pd.DataFrame(ties)
    tie_counts = tie_df.groupby("algorithm").size().rename("n_exact_ties").reset_index() if not tie_df.empty else pd.DataFrame(columns=["algorithm", "n_exact_ties"])
    eq_summary = eq_summary.merge(tie_counts, on="algorithm", how="outer").fillna(0)

    # Illustrative input-row exposure only; not CPU or elapsed-time equivalence.
    halving_units = 66 * 20 + 22 * 60 + 8 * 180 + 3 * 540 + 1620
    resources = pd.DataFrame([
        {"tuner": "Random Search / Bayesian / Evolutionary / Particle Swarm", "candidate_stage_events": 100,
         "resource_schedule_rows_per_event": "approximately full outer-training allocation",
         "illustrative_candidate_input_row_units": 100 * 800000, "excludes_final_refit": True,
         "note": "Approximate sample exposure; not CPU-work equivalence."},
        {"tuner": "Successive Halving", "candidate_stage_events": 100,
         "resource_schedule_rows_per_event": "66x20; 22x60; 8x180; 3x540; 1x1620",
         "illustrative_candidate_input_row_units": halving_units, "excludes_final_refit": True,
         "note": "Multi-fidelity sample exposure; not CPU-work equivalence."},
        {"tuner": "Approximate full-data-to-halving exposure ratio", "candidate_stage_events": np.nan,
         "resource_schedule_rows_per_event": "80000000 / 7320", "illustrative_candidate_input_row_units": 100 * 800000 / halving_units,
         "excludes_final_refit": True, "note": "Illustrative ratio only; not a runtime ratio."}])

    tables = {"summary_by_algorithm_tuner": combo, "summary_by_algorithm": by_algo,
        "ranked_balance_all_30": ranked, "pareto_frontier": pareto,
        "runtime_kendall_concordance": pd.DataFrame(concordance), "friedman_tests": pd.DataFrame(friedman),
        "friedman_average_ranks": pd.DataFrame(avg_ranks), "nemenyi_posthoc": pd.DataFrame(nemenyi),
        "corrected_paired_tost": tost, "equivalence_exact_ties": tie_df,
        "equivalence_zero_variance_exceptions": pd.DataFrame(zero_var, columns=["algorithm", "tuner", "reference_tuner", "n_splits", "mean_difference_tuner_minus_halving", "status"]),
        "equivalence_summary": eq_summary,
        "resource_budget": resources}
    if include_anova:
        work = data.copy()
        work["log_runtime"] = np.log10(work.runtime_sec)
        a, b, n = len(ALGORITHMS), len(TUNER_ORDER), CONFIG["outer_splits"]
        grand = work.log_runtime.mean()
        am = work.groupby("algorithm").log_runtime.mean().to_dict()
        tm = work.groupby("tuner").log_runtime.mean().to_dict()
        sm = work.groupby("split").log_runtime.mean().to_dict()
        cm = work.groupby(["algorithm", "tuner"]).log_runtime.mean().to_dict()
        ss_a = b*n*sum((am[x]-grand)**2 for x in ALGORITHMS)
        ss_t = a*n*sum((tm[x]-grand)**2 for x in TUNER_ORDER)
        ss_i = n*sum((cm[(x,y)]-am[x]-tm[y]+grand)**2 for x in ALGORITHMS for y in TUNER_ORDER)
        ss_s = a*b*sum((sm[x]-grand)**2 for x in range(1,n+1))
        ss_all = ((work.log_runtime-grand)**2).sum()
        ss_r = max(0, ss_all-ss_a-ss_t-ss_i-ss_s)
        dfs = [a-1, b-1, (a-1)*(b-1), n-1, len(work)-1-((a-1)+(b-1)+(a-1)*(b-1)+(n-1))]
        ss = [ss_a, ss_t, ss_i, ss_s, ss_r]
        names = ["Algorithm", "Tuner", "Algorithm x Tuner", "Split block", "Residual"]
        rows = []
        for name, value, df in zip(names, ss, dfs):
            f = p = np.nan
            if name != "Residual":
                f = (value/df)/(ss_r/dfs[-1]); p = stats.f.sf(f, df, dfs[-1])
            rows.append({"term": name, "sum_squares_log10_runtime": value, "df_numerator": df if name != "Residual" else np.nan,
                "df_denominator": dfs[-1] if name != "Residual" else np.nan, "F": f, "p_value": p,
                "eta_squared_term_over_total": value/ss_all})
        tables["runtime_anova_split_blocked_optional"] = pd.DataFrame(rows)

    for name, table in tables.items():
        table.to_csv(os.path.join(outdir, name + ".csv"), index=False, float_format="%.12g")
    with open(os.path.join(outdir, "analysis_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": len(data), "algorithms": ALGORITHMS, "tuners": TUNER_ORDER,
            "splits": sorted(data.split.unique().tolist()), "inner_cv_folds": CONFIG["inner_cv"],
            "nominal_candidate_stage_events": CONFIG["n_trials"], "runtime_anova_included": include_anova,
            "analysis_uses_recorded_values_only": True}, f, indent=2)
    print(f"Analysed {len(data)} recorded runs. Outputs saved to {os.path.abspath(outdir)}")
    print("No models were fitted; source workbook was not changed.")


def main():
    parser = argparse.ArgumentParser(description="Finance HPO benchmark and analysis script.")
    parser.add_argument("--mode", choices=["analyze", "benchmark"], default="analyze")
    parser.add_argument("--input", default="Complete data(2).xlsx")
    parser.add_argument("--sheet", default="0")
    parser.add_argument("--outdir", default="finance_hpo_analysis_outputs")
    parser.add_argument("--runtime-anova", choices=["none", "split-blocked"], default="none")
    args = parser.parse_args()
    if args.mode == "benchmark":
        X, y = load_finance()
        run_benchmark(X, y)
    else:
        sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        data = read_results(args.input, sheet)
        analyse_results(data, args.outdir, args.runtime_anova == "split-blocked")


if __name__ == "__main__":
    main()
