import pytest
from flow_grpo.consistency_schedule import consistency_weight


def test_noise_schedule_preserves_conditional_determinism_and_endpoints():
    assert consistency_weight(0, .5, "linear_alpha") == .5
    assert consistency_weight(1, .5, "linear_alpha") == 0
    assert consistency_weight(.8, .5, "linear_alpha") == pytest.approx(.1)
    assert consistency_weight(.8, .1) == .1


def test_schedule_rejects_invalid_coefficients():
    for sigma, amplitude in [(-.1,.5),(1.1,.5),(.5,1.1),(float("nan"),.5)]:
        with pytest.raises(ValueError):
            consistency_weight(sigma,amplitude)
    with pytest.raises(ValueError):
        consistency_weight(.5,.5,"ESS-dependent")
