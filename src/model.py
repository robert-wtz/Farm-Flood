"""
XGBoost flood probability model with spatial block cross-validation.
Block CV is required — random splits produce inflated AUC due to spatial autocorrelation.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def spatial_block_cv(df, n_blocks=5, lat_col="y", lon_col="x"):
    """
    Assign each pixel to a spatial block based on grid position.
    Returns array of block labels (0..n_blocks-1).
    """
    lat_bins = pd.cut(df[lat_col], bins=n_blocks, labels=False)
    lon_bins = pd.cut(df[lon_col], bins=n_blocks, labels=False)
    # Combine into block id
    blocks = lat_bins * n_blocks + lon_bins
    return blocks


def train_model(df, feature_cols, target_col="flooded", n_blocks=5,
                xgb_params=None, output_dir=None):
    """
    Train XGBoost with spatial block CV. Reports AUC and AP per fold.
    Returns trained model (fit on full data), CV metrics DataFrame.
    """
    xgb_params = xgb_params or {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42,
    }

    X = df[feature_cols].values
    y = df[target_col].values

    if "x" in df.columns and "y" in df.columns:
        blocks = spatial_block_cv(df, n_blocks=n_blocks)
    else:
        # Fallback: sequential blocks
        blocks = pd.cut(np.arange(len(df)), bins=n_blocks, labels=False)

    unique_blocks = sorted(set(blocks.dropna()))
    cv_results = []

    for test_block in unique_blocks:
        train_idx = blocks != test_block
        test_idx = blocks == test_block

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        cv_results.append({"block": test_block, "AUC": auc, "AP": ap,
                            "n_train": train_idx.sum(), "n_test": test_idx.sum()})
        print(f"Block {test_block}: AUC={auc:.3f}, AP={ap:.3f}")

    cv_df = pd.DataFrame(cv_results)
    print(f"\nMean AUC = {cv_df['AUC'].mean():.3f} ± {cv_df['AUC'].std():.3f}")
    print(f"Mean AP  = {cv_df['AP'].mean():.3f} ± {cv_df['AP'].std():.3f}")

    # Refit on full dataset
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X, y, verbose=False)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, output_dir / "model.joblib")
        cv_df.to_csv(output_dir / "cv_results.csv", index=False)

    return final_model, cv_df


def predict_raster(model, feature_raster_paths, output_path, feature_cols, ref_raster_path):
    """
    Run model on all pixels of input rasters → flood probability raster.
    """
    import rasterio
    import numpy as np

    with rasterio.open(ref_raster_path) as src:
        meta = src.meta.copy()
        shape = (src.height, src.width)

    arrays = []
    for col in feature_cols:
        path = feature_raster_paths[col]
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float).ravel()
            arrays.append(arr)

    X = np.column_stack(arrays)
    valid = ~np.any(np.isnan(X), axis=1)

    proba = np.full(X.shape[0], np.nan)
    proba[valid] = model.predict_proba(X[valid])[:, 1]
    proba_2d = proba.reshape(shape)

    meta.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(proba_2d.astype("float32"), 1)

    print(f"Probability raster saved: {output_path}")
