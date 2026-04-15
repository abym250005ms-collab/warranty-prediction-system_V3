from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import config
from data_preprocessing import ClaimPreprocessorArtifacts, fit_claim_preprocessor, transform_claim_features
from model_evaluation import evaluate_classification


@dataclass
class ClaimPredictionResult:
    metrics: Dict
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    artifacts: ClaimPreprocessorArtifacts



def _build_model(scale_pos_weight: float) -> XGBClassifier:
    params = dict(config.XGB_BASE_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def _feature_names(artifacts: ClaimPreprocessorArtifacts):
    return artifacts.preprocessor.get_feature_names_out()


def train_claim_model(df: pd.DataFrame) -> ClaimPredictionResult:
    target_col = "target_claim_next_3m"
    modeling_df = df.dropna(subset=[target_col]).copy()

    train_df, test_df = train_test_split(
        modeling_df,
        test_size=config.TEST_SIZE,
        stratify=modeling_df[target_col],
        random_state=config.RANDOM_STATE,
    )

    artifacts = fit_claim_preprocessor(train_df)
    X_train = transform_claim_features(train_df, artifacts)
    X_test = transform_claim_features(test_df, artifacts)
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = float(neg / max(pos, 1))

    model = _build_model(scale_pos_weight)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    tuner = GridSearchCV(
        estimator=model,
        param_grid=config.XGB_PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    tuner.fit(X_train, y_train)

    best_params = dict(config.XGB_BASE_PARAMS)
    best_params.update(tuner.best_params_)
    best_params["scale_pos_weight"] = scale_pos_weight

    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=config.RANDOM_STATE,
    )

    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train_main,
        y_train_main,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    test_prob = final_model.predict_proba(X_test)[:, 1]
    metrics = evaluate_classification(y_test, test_prob)
    metrics["best_params"] = tuner.best_params_
    metrics["cv_best_auc"] = float(tuner.best_score_)

    latest_vehicle_rows = modeling_df.sort_values("claim_date").groupby("vehicle_id", as_index=False).tail(1).copy()
    latest_vehicle_features = transform_claim_features(latest_vehicle_rows, artifacts)
    latest_vehicle_rows["risk_score"] = final_model.predict_proba(latest_vehicle_features)[:, 1]
    latest_vehicle_rows = latest_vehicle_rows.sort_values("risk_score", ascending=False)
    latest_vehicle_rows["risk_rank"] = np.arange(1, len(latest_vehicle_rows) + 1)

    prediction_cols = ["vehicle_id", "model_variant", "risk_score", "risk_rank"]
    predictions = latest_vehicle_rows[prediction_cols]

    importance = pd.DataFrame(
        {
            "feature": _feature_names(artifacts),
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config.CLAIM_MODEL_FILE)

    return ClaimPredictionResult(
        metrics=metrics,
        predictions=predictions,
        feature_importance=importance,
        artifacts=artifacts,
    )
