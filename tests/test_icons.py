"""The NumPy port of ICONS against outputs of the R package, and its basic behavior."""
import json
from pathlib import Path

import numpy as np
import pytest

from ggbfa._icons import detect, icons_paths, r_quantile

REFERENCE = json.loads((Path(__file__).parent / "data/icons_reference.json").read_text())


@pytest.mark.parametrize("case", range(len(REFERENCE["cases"])))
def test_paths_match_the_r_package(case):
    item = REFERENCE["cases"][case]
    ported = icons_paths(np.array(item["weights"]), item["percentiles"], item["lambdas"], item["min_size"])
    assert len(ported) == len(item["paths"])
    for saved, path in zip(item["paths"], ported):
        assert (saved["q"], saved["lambda"]) == (path["q"], path["lambda"])
        assert saved["threshold"] == path["threshold"]
        assert [list(map(int, b)) for b in saved["blocks"]] == path["blocks"]
        assert list(map(int, saved["background"])) == path["background"]


def test_r_quantile_is_type_7():
    rng = np.random.default_rng(3)
    values = rng.normal(size=101)
    for prob in (0, .1, .5, .93, 1):
        assert r_quantile(values, prob) == pytest.approx(np.quantile(values, prob), rel=1e-15, abs=1e-15)
    assert r_quantile([3.0, 1.0, 2.0], .5) == 2.0


def test_two_obvious_communities_and_background():
    w = np.zeros((9, 9))
    w[:4, :4] = .8
    w[4:7, 4:7] = .6
    np.fill_diagonal(w, 1)
    found = detect(w, threshold=.5, lam=.5, min_size=2)
    assert [sorted(b) for b in found["blocks"]] == [[0, 1, 2, 3], [4, 5, 6]]
    assert found["background"] == [7, 8]


def test_invalid_input():
    with pytest.raises(ValueError):
        icons_paths(np.ones((3, 2)), [90], [.5])
    with pytest.raises(ValueError):
        icons_paths(np.array([[1, .2], [.3, 1]]), [90], [.5])
    with pytest.raises(ValueError):
        icons_paths(np.eye(3), [90], [.5], min_size=1)
