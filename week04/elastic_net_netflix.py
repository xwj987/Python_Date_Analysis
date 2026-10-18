"""Week 4: Elastic Net analysis of Kaggle's Netflix Titles data.

The script accepts a local netflix_titles.csv as its first argument. If none is
provided, it downloads the same public Kaggle dataset mirror used for the
reproducible example.
"""
from pathlib import Path
import sys
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path(__file__).parent
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/shivamb/netflix-shows"

def load_data():
    if len(sys.argv) > 1:
        return pd.read_csv(sys.argv[1]), Path(sys.argv[1]).name
    payload = urllib.request.urlopen(KAGGLE_URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith("netflix_titles.csv"))
        return pd.read_csv(archive.open(csv_name)), "Kaggle: shivamb/netflix-shows"

def main():
    df, source = load_data()
    # Duration is a usable continuous response after removing TV shows and
    # converting strings such as "90 min" to minutes.
    data = df[df["type"].eq("Movie")].copy()
    data["duration_min"] = pd.to_numeric(data["duration"].str.extract(r"(\d+)")[0], errors="coerce")
    data["date_added"] = pd.to_datetime(data["date_added"], errors="coerce")
    data["added_year"] = data["date_added"].dt.year
    data["added_month"] = data["date_added"].dt.month
    data = data.dropna(subset=["duration_min"])
    features = ["release_year", "added_year", "added_month", "rating", "country", "listed_in"]
    X, y = data[features], data["duration_min"]
    numeric = ["release_year", "added_year", "added_month"]
    categorical = ["rating", "country", "listed_in"]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = Pipeline([("preprocess", preprocess),
                      ("elastic_net", ElasticNetCV(
                          l1_ratio=np.linspace(.05, 1, 20),
                          alphas=np.logspace(-3, 2, 80), cv=10,
                          max_iter=100000, random_state=42))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    en = model.named_steps["elastic_net"]
    metrics = pd.DataFrame([{
        "source": source, "rows_used": len(data), "alpha": en.alpha_,
        "l1_ratio": en.l1_ratio_, "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "test_mae": mean_absolute_error(y_test, pred), "test_r2": r2_score(y_test, pred),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(en.coef_) > 1e-8)),
    }])
    metrics.to_csv(OUT / "metrics.csv", index=False)
    plt.figure(figsize=(6, 5)); plt.scatter(y_test, pred, alpha=.35)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("actual movie duration (minutes)"); plt.ylabel("predicted duration (minutes)")
    plt.title("Netflix movie duration: Elastic Net"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=180)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
