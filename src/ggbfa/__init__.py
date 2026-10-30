"""gg-BFA: graph-guided Bayesian factor analysis.

``efa`` learns the communities of the variables and their number (the
exploratory step); ``cfa`` fits one of the four confirmatory members (1f, 2f+s,
2f, mf) to them (the confirmatory step).
"""

from ._version import __version__
from ._backend import GPUBackendUnavailable
from ._config import CFAConfig
from ._members import MEMBERS
from ._cfa import CFAFit, cfa
from ._efa import EFAResult, efa
from ._bagging import BaggedCFAFit
from ._joint import JointCovariance
from ._posterior import PosteriorSummary
from ._prediction import MaskedPrediction



def __getattr__(name):
    if name == "plot":  # imported on first use: matplotlib is an optional dependency
        import importlib

        return importlib.import_module(".plot", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaggedCFAFit",
    "CFAConfig",
    "CFAFit",
    "EFAResult",
    "GPUBackendUnavailable",
    "JointCovariance",
    "MEMBERS",
    "MaskedPrediction",
    "PosteriorSummary",
    "__version__",
    "cfa",
    "efa",
]
