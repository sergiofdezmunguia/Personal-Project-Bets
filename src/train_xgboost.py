"""Train a first betting model (Over 2.5 goals) with temporal validation.

Features are generated pre-match from rolling team form to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.config import MODELS_DIR, RAW_DATA_DIR


@dataclass
class DatasetBundle:
    train_x: pd.DataFrame
    train_y: pd.Series
    valid_x: pd.DataFrame
    valid_y: pd.Series
    test_x: pd.DataFrame
    test_y: pd.Series


def load_raw_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule_path = RAW_DATA_DIR / "understat_schedule.csv"
    team_stats_path = RAW_DATA_DIR / "understat_team_match_stats.csv"

    if not schedule_path.exists() or not team_stats_path.exists():
        raise FileNotFoundError(
            "Missing input CSVs. Run src/extract_pipeline.py or src/explore_understat.py first."
        )

    schedule = pd.read_csv(schedule_path)
    team_stats = pd.read_csv(team_stats_path)
    return schedule, team_stats


def prepare_base_table(schedule: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    # Flatten index columns that were saved by pandas from a multi-index DataFrame.
    drop_cols = [c for c in ["league", "season", "game"] if c in schedule.columns]
    if drop_cols:
        schedule = schedule.drop(columns=drop_cols)

    drop_cols_stats = [c for c in ["league", "season", "game"] if c in team_stats.columns]
    if drop_cols_stats:
        team_stats = team_stats.drop(columns=drop_cols_stats)

    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
    team_stats["date"] = pd.to_datetime(team_stats["date"], errors="coerce")

    key_cols = [
        "league_id",
        "season_id",
        "game_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    model_df = schedule[key_cols].copy()

    model_df["total_goals"] = model_df["home_goals"] + model_df["away_goals"]
    model_df["target_over_2_5"] = (model_df["total_goals"] > 2.5).astype(int)

    # Keep only completed matches.
    model_df = model_df.dropna(subset=["date", "home_goals", "away_goals"])

    # Include tactical/team-level match stats for feature engineering.
    merged = model_df.merge(
        team_stats,
        on=["league_id", "season_id", "game_id", "date", "home_team", "away_team"],
        how="inner",
    )
    return merged


def build_team_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    home_view = pd.DataFrame(
        {
            "league_id": df["league_id"],
            "season_id": df["season_id"],
            "game_id": df["game_id"],
            "date": df["date"],
            "team": df["home_team"],
            "xg_for": df["home_xg"],
            "xg_against": df["away_xg"],
            "points": df["home_points"],
            "ppda": df["home_ppda"],
            "deep_comp": df["home_deep_completions"],
            "side": "home",
        }
    )

    away_view = pd.DataFrame(
        {
            "league_id": df["league_id"],
            "season_id": df["season_id"],
            "game_id": df["game_id"],
            "date": df["date"],
            "team": df["away_team"],
            "xg_for": df["away_xg"],
            "xg_against": df["home_xg"],
            "points": df["away_points"],
            "ppda": df["away_ppda"],
            "deep_comp": df["away_deep_completions"],
            "side": "away",
        }
    )

    long_df = pd.concat([home_view, away_view], ignore_index=True)
    long_df = long_df.sort_values(["team", "date", "game_id"]).reset_index(drop=True)

    metrics = ["xg_for", "xg_against", "points", "ppda", "deep_comp"]
    for metric in metrics:
        long_df[f"{metric}_roll{window}"] = (
            long_df.groupby("team")[metric].shift(1).rolling(window, min_periods=3).mean()
        )

    keep_cols = [
        "league_id",
        "season_id",
        "game_id",
        "side",
        "xg_for_roll5",
        "xg_against_roll5",
        "points_roll5",
        "ppda_roll5",
        "deep_comp_roll5",
    ]
    feat_long = long_df[keep_cols]

    home_feats = feat_long[feat_long["side"] == "home"].drop(columns=["side"])
    home_feats = home_feats.rename(
        columns={
            "xg_for_roll5": "home_xg_for_roll5",
            "xg_against_roll5": "home_xg_against_roll5",
            "points_roll5": "home_points_roll5",
            "ppda_roll5": "home_ppda_roll5",
            "deep_comp_roll5": "home_deep_comp_roll5",
        }
    )

    away_feats = feat_long[feat_long["side"] == "away"].drop(columns=["side"])
    away_feats = away_feats.rename(
        columns={
            "xg_for_roll5": "away_xg_for_roll5",
            "xg_against_roll5": "away_xg_against_roll5",
            "points_roll5": "away_points_roll5",
            "ppda_roll5": "away_ppda_roll5",
            "deep_comp_roll5": "away_deep_comp_roll5",
        }
    )

    out = df.merge(home_feats, on=["league_id", "season_id", "game_id"], how="left")
    out = out.merge(away_feats, on=["league_id", "season_id", "game_id"], how="left")

    out["roll_xg_for_diff"] = out["home_xg_for_roll5"] - out["away_xg_for_roll5"]
    out["roll_points_diff"] = out["home_points_roll5"] - out["away_points_roll5"]
    out["roll_ppda_diff"] = out["home_ppda_roll5"] - out["away_ppda_roll5"]

    return out


def temporal_split(df: pd.DataFrame) -> DatasetBundle:
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [
        "home_xg_for_roll5",
        "home_xg_against_roll5",
        "home_points_roll5",
        "home_ppda_roll5",
        "home_deep_comp_roll5",
        "away_xg_for_roll5",
        "away_xg_against_roll5",
        "away_points_roll5",
        "away_ppda_roll5",
        "away_deep_comp_roll5",
        "roll_xg_for_diff",
        "roll_points_diff",
        "roll_ppda_diff",
    ]

    data = df.dropna(subset=feature_cols + ["target_over_2_5"]).copy()

    # 70/15/15 temporal split by row order after sorting by date.
    n = len(data)
    i_train = int(n * 0.70)
    i_valid = int(n * 0.85)

    train = data.iloc[:i_train]
    valid = data.iloc[i_train:i_valid]
    test = data.iloc[i_valid:]

    return DatasetBundle(
        train_x=train[feature_cols],
        train_y=train["target_over_2_5"],
        valid_x=valid[feature_cols],
        valid_y=valid["target_over_2_5"],
        test_x=test[feature_cols],
        test_y=test["target_over_2_5"],
    )


def train_model(bundle: DatasetBundle):
    # Prefer XGBoost when available; fallback keeps script runnable.
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="binary:logistic",
            eval_metric="logloss",
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(random_state=42)

    model.fit(bundle.train_x, bundle.train_y)
    return model


def evaluate(model, x: pd.DataFrame, y: pd.Series, split_name: str) -> dict[str, float]:
    proba = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "logloss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
    }
    print(f"[{split_name}] {metrics}")
    return metrics


def main() -> None:
    schedule, team_stats = load_raw_inputs()
    base_df = prepare_base_table(schedule, team_stats)
    feat_df = build_team_form_features(base_df, window=5)
    bundle = temporal_split(feat_df)

    model = train_model(bundle)

    metrics = {
        "valid": evaluate(model, bundle.valid_x, bundle.valid_y, "valid"),
        "test": evaluate(model, bundle.test_x, bundle.test_y, "test"),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgboost_over25.joblib"
    metrics_path = MODELS_DIR / "xgboost_over25_metrics.json"

    joblib.dump(model, model_path)
    pd.Series(metrics).to_json(metrics_path, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
