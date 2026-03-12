import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels


# Columns that are metadata, not features
_META_COLS = {"Season", "ATeamName", "BTeamName", "AWon"}


def split_n_scale(df, test_seasons=None, test_size=0.2, random_state=42):
    """
    Prepare train/test splits from a matchup DataFrame.

    Columns ending in '_A' or '_B' are treated as features.
    'AWon' is the target (bool/int: True/1 = Team A won).

    Parameters
    ----------
    df : pd.DataFrame
        Raw matchup data. Must contain 'AWon' and feature columns
        named '<FeatureName>_A' / '<FeatureName>_B'.
    test_seasons : list[int] or None
        If provided, rows whose 'Season' is in this list become the
        test set. Otherwise a random stratified split is used.
    test_size : float
        Fraction used for the random split when test_seasons is None.
    random_state : int
        Seed for the random split.

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame
        Scaled feature matrices (StandardScaler fitted on train only).
    y_train, y_test : pd.Series
        Binary targets (1 = Team A won).
    train_df, test_df : pd.DataFrame
        Raw (unscaled) subsets of df for reference.
    """
    # Target
    y = df["AWon"].astype(int)

    # Feature columns: anything ending in _A or _B that isn't metadata
    feature_cols = [
        c for c in df.columns
        if (c.endswith("_A") or c.endswith("_B")) and c not in _META_COLS
    ]

    X = df[feature_cols].copy()

    # --- train / test split ---
    if test_seasons is not None and "Season" in df.columns:
        test_mask = df["Season"].isin(test_seasons)
    else:
        from sklearn.model_selection import train_test_split
        idx_train, idx_test = train_test_split(
            df.index, test_size=test_size, random_state=random_state,
            stratify=y
        )
        test_mask = df.index.isin(idx_test)

    train_mask = ~test_mask

    X_train_raw = X.loc[train_mask].reset_index(drop=True)
    X_test_raw  = X.loc[test_mask].reset_index(drop=True)
    y_train     = y.loc[train_mask].reset_index(drop=True)
    y_test      = y.loc[test_mask].reset_index(drop=True)
    train_df    = df.loc[train_mask].reset_index(drop=True)
    test_df     = df.loc[test_mask].reset_index(drop=True)

    # --- scale (fit on train only) ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=feature_cols
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, train_df, test_df


def lasso_cols(X_train_scaled, y_train, alpha):
    """
    Run LASSO to select base features.
    Returns (base_feature_names, full_column_names).
    Expects columns named like 'FeatureName_A' / 'FeatureName_B'.
    """
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    selected_cols = [
        col for col, coef in zip(X_train_scaled.columns, lasso.coef_)
        if coef != 0
    ]

    # Strip _A / _B suffix to get base feature names
    base_features = sorted({
        col.rsplit("_", 1)[0]
        for col in selected_cols
        if col.endswith("_A") or col.endswith("_B")
    })

    return base_features, selected_cols


class MixtureOfExperts(BaseEstimator, ClassifierMixin):
    """
    Mixture-of-Experts classifier compatible with scikit-learn API.

    Each expert is a logistic regression trained on a random subset of
    LASSO-selected base features (team-paired as <feat>_A / <feat>_B).
    A meta logistic regression learns mixture weights over the experts.

    Parameters
    ----------
    alpha : float
        LASSO regularisation strength for feature selection.
    n_features : int
        Number of base features sampled per expert.
    n_experts : int
        Number of expert logistic regressions to train.
    C_expert : float
        Inverse regularisation strength for each expert model.
    C_meta : float
        Inverse regularisation strength for the meta model.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        n_features: int = 5,
        n_experts: int = 20,
        C_expert: float = 1.0,
        C_meta: float = 10.0,
        random_state=None,
    ):
        self.alpha = alpha
        self.n_features = n_features
        self.n_experts = n_experts
        self.C_expert = C_expert
        self.C_meta = C_meta
        self.random_state = random_state

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the mixture of experts.

        Parameters
        ----------
        X : pd.DataFrame
            Training features. Columns must include paired names like
            'SomeFeature_A' and 'SomeFeature_B'.
        y : array-like of shape (n_samples,)
            Binary target labels.
        """
        X = self._validate_dataframe(X)
        self.classes_ = unique_labels(y)
        self.feature_names_in_ = np.array(X.columns)

        rng = np.random.default_rng(self.random_state)

        # ---- 1. LASSO feature selection --------------------------------
        base_features, _ = lasso_cols(X, y, self.alpha)
        base_features = sorted(set(base_features))
        self.selected_base_features_ = base_features

        # ---- 2. Train expert models ------------------------------------
        train_probs = []
        self.experts_ = []
        self.expert_feature_sets_ = []

        for _ in range(self.n_experts):
            chosen = rng.choice(
                base_features,
                size=min(self.n_features, len(base_features)),
                replace=False,
            )

            expert_cols = [
                col
                for f in chosen
                for col in (f"{f}_A", f"{f}_B")
                if col in X.columns
            ]

            if len(expert_cols) == 0:
                continue

            expert = LogisticRegression(
                penalty="l2",
                C=self.C_expert,
                solver="lbfgs",
                max_iter=3000,
            )
            expert.fit(X[expert_cols], y)

            probs = expert.predict_proba(X[expert_cols])[:, 1]
            train_probs.append(probs)
            self.experts_.append(expert)
            self.expert_feature_sets_.append(expert_cols)

        if len(train_probs) == 0:
            raise ValueError(
                "No experts were built — check that X contains paired "
                "columns ending in '_A' and '_B' and that LASSO selects "
                "at least one feature."
            )

        # ---- 3. Meta model (learn mixture weights) ---------------------
        Z_train = np.column_stack(train_probs)

        self.meta_model_ = LogisticRegression(
            penalty="l2",
            C=self.C_meta,
            solver="lbfgs",
            max_iter=3000,
        )
        self.meta_model_.fit(Z_train, y)

        # Normalised non-negative weights
        raw_weights = np.maximum(self.meta_model_.coef_.ravel(), 0)
        total = raw_weights.sum()
        self.weights_ = raw_weights / total if total > 0 else np.ones(len(raw_weights)) / len(raw_weights)

        return self

    def predict_proba(self, X):
        """
        Return class probabilities.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
        """
        check_is_fitted(self)
        X = self._validate_dataframe(X)

        expert_probs = [
            expert.predict_proba(X[cols])[:, 1]
            for expert, cols in zip(self.experts_, self.expert_feature_sets_)
        ]
        Z = np.column_stack(expert_probs)
        p = Z @ self.weights_
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Predict binary class labels (threshold = 0.5)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-15, 1))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def weights_summary(self, top_n: int = 10) -> pd.DataFrame:
        """Return a tidy DataFrame of expert weights."""
        check_is_fitted(self)
        return (
            pd.DataFrame({
                "Expert": [f"Expert_{i}" for i in range(len(self.weights_))],
                "Weight": self.weights_,
                "Features": [", ".join(cols) for cols in self.expert_feature_sets_],
            })
            .sort_values("Weight", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "MixtureOfExperts requires a pd.DataFrame with column names "
                "ending in '_A' / '_B' (e.g. 'Seed_A', 'Seed_B')."
            )
        return X
