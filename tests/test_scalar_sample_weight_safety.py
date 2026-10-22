import numpy as np
import pytest
from bayesbreak.validation import check_sample_weight

@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_scalar_weights_rejected(bad):
    with pytest.raises(ValueError, match="sample_weight"):
        check_sample_weight(bad, 5)

def test_scalar_weights_broadcast():
    np.testing.assert_array_equal(check_sample_weight(2.0, 3), [2.0, 2.0, 2.0])
