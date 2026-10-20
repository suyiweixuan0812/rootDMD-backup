import pytest
import torch
from flow_grpo.diffusers_patch.sd3_last_innovation_path_is import cached_prefix_fields


def test_cached_prefix_fields_are_exact_and_require_identical_parents():
    calls = []
    def field(x, s):
        calls.append((len(x), s))
        return (x*2+s).to(x.dtype)
    states = [torch.randn(1, 3, 4, dtype=torch.bfloat16) for _ in range(3)]
    a, b, record = cached_prefix_fields(field, field, states, [1., .9, .5])
    for _ in range(2):
        x = states[1].expand(8, -1, -1)
        assert torch.equal(a(x, .9), field(x, .9))
        assert torch.equal(b(x, .9), field(x, .9))
    assert record['student_network_calls'] == record['teacher_network_calls'] == 1
    assert record['cached_field_hits'] == 2
    with pytest.raises(RuntimeError):
        a(states[1]+1, .9)
    # Last readout is variable and must always call the actual field.
    x = states[2]+1
    assert torch.equal(a(x, .5), field(x, .5))
