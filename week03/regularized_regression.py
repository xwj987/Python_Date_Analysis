"""Week 3, exercise 4: regularized regression on Hitters."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv"

def main():
    df = pd.read_csv(DATA_URL).dropna()
    target = "Salary"
    X = pd.get_dummies(df.drop(columns=target), drop_first=True, dtype=float)
    y = df[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    alphas = np.logspace(-4, 4, 160)
    models = {
        "Ridge": RidgeCV(alphas=alphas, cv=10),
        "Lasso": LassoCV(alphas=alphas, cv=10, max_iter=100000, random_state=42),
        "ElasticNet": ElasticNetCV(
            alphas=alphas, l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0],
            cv=10, max_iter=100000, random_state=42
        ),
    }
    rows, paths = [], {}
    for name, estimator in models.items():
        pipe = make_pipeline(StandardScaler(), estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        coef = estimator.coef_
        rows.append({"model": name, "alpha": estimator.alpha_,
                     "l1_ratio": getattr(estimator, "l1_ratio_", np.nan),
                     "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
                     "nonzero": int(np.count_nonzero(np.abs(coef) > 1e-8))})
        paths[name] = (estimator, coef)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (name, (estimator, _)) in zip(axes, paths.items()):
        # Refit along an alpha grid to show the complete coefficient path.
        path_class = {"Ridge": Ridge, "Lasso": Lasso, "ElasticNet": ElasticNet}[name]
        path_alphas = np.logspace(-3, 3, 70)
        path_coefs = []
        for alpha in path_alphas:
            clone = make_pipeline(StandardScaler(), path_class(alpha=alpha, **(
                {"l1_ratio": estimator.l1_ratio_} if name == "ElasticNet" else
                {"max_iter": 100000} if name == "Lasso" else {})))
            clone.fit(X_train, y_train)
            path_coefs.append(clone[-1].coef_)
        for coef in np.asarray(path_coefs).T:
            ax.plot(path_alphas, coef, color="steelblue", alpha=.35)
        ax.axvline(estimator.alpha_, color="crimson", linestyle="--", label="CV alpha")
        ax.set_xscale("log"); ax.set_title(name); ax.set_xlabel("alpha")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("standardized coefficient")
    axes[-1].legend()
    fig.suptitle("Hitters regularization coefficient paths")
    fig.tight_layout(); fig.savefig(OUT / "coefficient_paths.png", dpi=180)
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
