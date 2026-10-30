# ggbfa: graph-guided Bayesian factor analysis

`ggbfa` implements gg-BFA, a fast integrated EFA–CFA approach for
high-dimensional data, including data with more variables than subjects.

- **Exploratory step (`efa`).** Learns which variables share a factor, how many
  factors there are, and which variables belong to no factor. A dense-subgraph
  search proposes nested paths of communities. A forward stepwise search then
  compares fractional variational posteriors and stops when the ELBO first
  decreases.
- **Confirmatory step (`cfa`).** Fits one of four Bayesian factor models to the
  communities. The result gives posterior inference for the loadings, the
  factors and their covariance, and predicts missing values.

## Installation

```bash
pip install "ggbfa[plot] @ git+https://github.com/waterism211/ggbfa"
```

The package needs NumPy and SciPy. The `plot` extra adds matplotlib. The `gpu`
extra adds CuPy for CUDA 12, which runs the variational updates on a GPU.

## Quick start

```python
import ggbfa
from ggbfa import plot

res = ggbfa.efa(X)                      # exploratory step; X is n x p
res.n_factors, res.communities          # number of communities and their variables
fit = ggbfa.cfa(X, res, member="2f+s")  # confirmatory step

fit.covariance()                        # E_q[Lambda Phi Lambda' + Psi], p x p
fit.loadings_, fit.cross_loading_pip_   # posterior mean loadings and inclusion probabilities
fit.factor_covariance_, fit.factor_covariance_ci_
fit.factor_scores_
fit.sample_posterior(1000)              # joint draws of F, Lambda, gamma, psi and Phi
fit.predict_posterior(row, masked)      # predict the masked entries of a new subject

plot.plot_search(res)                   # early stopping of the search
plot.plot_correlation(res, X, fit)      # sample (upper) and fitted (lower) correlation
plot.plot_factor_covariance(fit)        # marginal posteriors of Phi with credible intervals
plot.plot_factor_network(fit)           # strong factor correlations
plot.plot_loadings(fit)
```

If the communities are already known, pass them directly to `cfa` as a list of
0-based column-index arrays.

## Confirmatory members

| member | loadings of each variable |
|---|---|
| `"1f"` | its own community only; loadings fixed; residual variance shared within the community |
| `"2f+s"` | at most one cross-loading, and at most a fraction `max_overlap_fraction` of each community has one |
| `"2f"` | at most one cross-loading |
| `"mf"` | spike-and-slab cross-loadings on every other factor |

By default, the loadings of each variable sum to one, which fixes the scale of
each factor. With `loading_normalization="free"`, the primary loadings are
estimated under a normal prior, and only that prior fixes the scales.

## Options

- `efa(X, search="two_stage")`: the two-stage search used for the real-data
  analyses of the paper. `search="full"` compares every candidate partition at
  each number of communities, as analyzed by the theory.
- `cfa(X, communities, member=..., bagging=20)`: variational bagging with 20
  bootstrap fits of 2n rows each.
- `device="auto"`, `"cpu"` or `"gpu"`. `"auto"` uses a GPU when CuPy finds one
  and records the choice in `fit.metadata_["runtime"]`. `"gpu"` fails instead
  of falling back to the CPU.
- Prior and optimization settings are keyword arguments of `efa` and `cfa`.
  For `cfa`, they are the fields of `ggbfa.CFAConfig`. The defaults are those
  of the paper.

## Reproducing the paper

The analyses, simulations and figures of the paper, with the exact
configurations and saved outputs, are in
<https://github.com/waterism211/bayesian_cfa>.

## Acknowledgement

The dense-subgraph search is a NumPy port of the R package ICONS by Yezhi Pan
(<https://github.com/xavienzo/ICONS>, MIT license; see `LICENSES/ICONS.txt`).
It returns the same communities as the R package.

## License

MIT; see `LICENSE`.
