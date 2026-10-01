"""Probit with average marginal effects (optionally survey-weighted) and a recursive bivariate probit."""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tools.numdiff import approx_fprime

_PHI = stats.norm.cdf
_phi = stats.norm.pdf


# ---------------------------------------------------------------- probit + AME
def fit_probit(df, y, rhs, weight=None):
    """Probit of y on rhs (patsy formula RHS). Robust (HC1-style sandwich) SEs.

    weight: column name of survey weights (rescaled to mean 1) or None.
    Returns dict with result, design info and the AME table (delta-method SEs).
    """
    cols = [y] + ([weight] if weight else [])
    data = df.copy()
    yv, X = patsy.dmatrices(f"{y} ~ {rhs}", data, return_type="dataframe", NA_action="drop")
    idx = X.index
    w = None
    if weight:
        w = data.loc[idx, weight].astype(float)
        w = w / w.mean()
    mod = sm.GLM(yv.iloc[:, 0], X, family=sm.families.Binomial(link=sm.families.links.Probit()),
                 freq_weights=w if w is not None else None)
    res = mod.fit(cov_type="HC0" if w is not None else "HC1")
    ame = _ame_table(res, X, w, yv.columns[0])
    return {"res": res, "X": X, "n": len(idx), "ame": ame}


def _ame_from_params(params, X, w, is_bin):
    xb = X.values @ params
    out = []
    wts = np.ones(len(X)) if w is None else w.values
    for j, col in enumerate(X.columns):
        if col == "Intercept":
            out.append(np.nan)
            continue
        if is_bin[j]:
            x1, x0 = X.values.copy(), X.values.copy()
            x1[:, j], x0[:, j] = 1, 0
            eff = _PHI(x1 @ params) - _PHI(x0 @ params)
        else:
            eff = _phi(xb) * params[j]
        out.append(np.average(eff, weights=wts))
    return np.array(out)


def _ame_table(res, X, w, yname):
    is_bin = [set(np.unique(X[c])) <= {0.0, 1.0} for c in X.columns]
    params = res.params.values
    est = _ame_from_params(params, X, w, is_bin)
    jac = approx_fprime(params, lambda p: _ame_from_params(p, X, w, is_bin), centered=True)
    cov = res.cov_params().values
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", jac, cov, jac), 0, None))
    t = est / se
    tab = pd.DataFrame({"ame": est, "se": se, "z": t, "p": 2 * (1 - _PHI(np.abs(t)))}, index=X.columns)
    return tab.drop(index="Intercept")


def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def fmt_ame(tab, keep=None):
    t = tab if keep is None else tab.loc[[k for k in keep if k in tab.index]]
    return t.assign(ame_fmt=lambda d: d.ame.map("{:+.3f}".format) + d.p.map(stars),
                    se_fmt=lambda d: "(" + d.se.map("{:.3f}".format) + ")")


# ------------------------------------------------- recursive bivariate probit
def _bvn_cdf(h, k, rho, n_nodes=64):
    """P(Z1<=h, Z2<=k) for standard bivariate normal with correlation rho (array h,k; scalar rho).
    Phi2 = Phi(h)Phi(k) + integral_0^rho phi2(h,k;t) dt, by Gauss-Legendre (accurate for |rho|<~0.97)."""
    x, wq = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * rho * (x + 1.0)              # nodes on [0, rho]
    wt = 0.5 * rho * wq
    h = np.asarray(h)[:, None]
    k = np.asarray(k)[:, None]
    dens = np.exp(-(h ** 2 - 2 * t * h * k + k ** 2) / (2 * (1 - t ** 2))) / (2 * np.pi * np.sqrt(1 - t ** 2))
    return _PHI(h[:, 0]) * _PHI(k[:, 0]) + dens @ wt


class RecursiveBiprobit(GenericLikelihoodModel):
    """y1 = 1[ z'g + u > 0 ]  (endogenous regressor equation, e.g. innovation)
       y2 = 1[ a*y1 + x'b + e > 0 ]  (outcome equation, e.g. exports);  corr(u,e) = rho.
    Params: g (kz), [a, b] (1+kx), atanh(rho)."""

    def __init__(self, y1, Z, y2, X, **kw):
        self.y1, self.y2 = np.asarray(y1, float), np.asarray(y2, float)
        self.Z, self.X = np.asarray(Z, float), np.asarray(X, float)
        self.kz, self.kx = self.Z.shape[1], self.X.shape[1]
        super().__init__(endog=self.y2, exog=self.X, **kw)
        self.nparams = self.kz + 1 + self.kx + 1

    def loglikeobs(self, p):
        g = p[: self.kz]
        a = p[self.kz]
        b = p[self.kz + 1: self.kz + 1 + self.kx]
        rho = np.tanh(p[-1])
        rho = np.clip(rho, -0.97, 0.97)
        q1, q2 = 2 * self.y1 - 1, 2 * self.y2 - 1
        u = self.Z @ g
        v = a * self.y1 + self.X @ b
        ll = np.empty(len(q1))
        for s in (-1.0, 1.0):              # observations share rho*s within each sign group
            m = (q1 * q2) == s
            if m.any():
                pr = _bvn_cdf(q1[m] * u[m], q2[m] * v[m], s * rho)
                ll[m] = np.log(np.clip(pr, 1e-300, None))
        return ll

    def fit_model(self, start=None, **kw):
        if start is None:
            start = np.zeros(self.nparams)
            g0 = sm.Probit(self.y1, self.Z).fit(disp=0).params
            xb = np.column_stack([self.y1, self.X])
            b0 = sm.Probit(self.y2, xb).fit(disp=0).params
            start = np.concatenate([g0, b0, [0.0]])
        return self.fit(start_params=start, method="bfgs", maxiter=500, disp=0, **kw)


def fit_biprobit(df, y_innov, y_out, z_terms, x_terms):
    """z_terms / x_terms: patsy RHS strings for the innovation and outcome equations (no intercept needed)."""
    used = [y_innov, y_out]
    Zm = patsy.dmatrix(z_terms, df, return_type="dataframe", NA_action="drop")
    Xm = patsy.dmatrix(x_terms, df, return_type="dataframe", NA_action="drop")
    idx = Zm.index.intersection(Xm.index).intersection(df[used].dropna().index)
    Zm, Xm, d = Zm.loc[idx], Xm.loc[idx], df.loc[idx]
    mod = RecursiveBiprobit(d[y_innov], Zm, d[y_out], Xm)
    res = mod.fit_model()
    kz, kx = Zm.shape[1], Xm.shape[1]
    p = res.params
    names = [f"innov_eq:{c}" for c in Zm.columns] + ["export_eq:innov"] + [f"export_eq:{c}" for c in Xm.columns] + ["atanh_rho"]
    tab = pd.DataFrame({"coef": p, "se": res.bse, "p": res.pvalues}, index=names)
    alpha = p[kz]
    b = p[kz + 1: kz + 1 + kx]
    xb = Xm.values @ b
    ate = float(np.mean(_PHI(xb + alpha) - _PHI(xb)))       # average effect of innovating on P(export)

    def ate_fn(pp):
        return np.array([np.mean(_PHI(Xm.values @ pp[kz + 1: kz + 1 + kx] + pp[kz]) - _PHI(Xm.values @ pp[kz + 1: kz + 1 + kx]))])

    jac = np.atleast_2d(approx_fprime(np.asarray(p, float), ate_fn, centered=True)).reshape(1, -1)
    cov = res.cov_params()
    cov = cov.values if hasattr(cov, "values") else cov
    ate_se = float(np.sqrt(max(float((jac @ cov @ jac.T).ravel()[0]), 0)))
    rho = float(np.tanh(p.iloc[-1] if hasattr(p, "iloc") else p[-1]))
    # Wald test of rho=0 (exogeneity of innovation)
    wald_p = float(tab.loc["atanh_rho", "p"])
    return {"table": tab, "ate": ate, "ate_se": ate_se, "rho": rho, "rho_p": wald_p, "n": len(idx),
            "converged": bool(res.mle_retvals.get("converged", True)), "llf": float(res.llf)}
