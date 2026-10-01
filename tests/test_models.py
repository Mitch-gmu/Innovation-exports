import sys
from pathlib import Path
import numpy as np, pandas as pd
import pytest
from scipy.stats import multivariate_normal as mvn
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import models


@pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.5, 0.9])
def test_bivariate_normal_cdf_matches_scipy(rho):
    rng = np.random.default_rng(1)
    h, k = rng.normal(size=8) * 1.5, rng.normal(size=8) * 1.5
    ref = np.array([mvn.cdf([a, b], cov=[[1, rho], [rho, 1]]) for a, b in zip(h, k)])
    assert np.abs(models._bvn_cdf(h, k, rho) - ref).max() < 1e-6


def test_ame_matches_statsmodels():
    rng = np.random.default_rng(2)
    n = 1500
    df = pd.DataFrame({"x": rng.integers(0, 2, n), "z": rng.normal(size=n)})
    df["y"] = (0.4 * df.x + 0.3 * df.z + rng.normal(size=n) > 0.2).astype(float)
    mine = models.fit_probit(df, "y", "x + z")["ame"]
    ref = smf.probit("y ~ x + z", df).fit(disp=0, cov_type="HC1").get_margeff(at="overall", dummy=True).summary_frame()
    assert np.allclose(mine.loc["x", "ame"], ref.loc["x", "dy/dx"], atol=1e-6)
    assert np.allclose(mine.loc["z", "ame"], ref.loc["z", "dy/dx"], atol=1e-6)


def test_biprobit_recovers_positive_effect():
    rng = np.random.default_rng(3)
    n = 3000
    df = pd.DataFrame({"x": rng.normal(size=n), "iv": rng.normal(size=n)})
    u, e = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], n).T
    df["d"] = (0.3 * df.x + 0.8 * df.iv + u > 0).astype(float)
    df["y"] = (0.8 * df.d + 0.3 * df.x + e > 0.2).astype(float)
    r = models.fit_biprobit(df, "d", "y", "x + iv", "x")
    assert r["converged"] and r["ate"] > 0.1
