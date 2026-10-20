"""Last positive innovation block, exact cached prefix fields, pure Teacher map."""
import torch
from .sd3_random_walk_path_is import conditional_path_mean as random_walk_mean
from .sd3_conditional_path_is import make_fields, sample_prefix


def cached_prefix_fields(student, teacher, states, positive_sigmas):
    """Only unchanged prefix parents can use cache; any mismatch fails loudly.

    Calls are batch1 before expanding the exact result, matching field_microbatch1.
    The current model parameters must stay fixed for the lifetime of this cache.
    """
    parents = {float(positive_sigmas[k]): states[k] for k in range(len(states)-1)}
    values = {}
    calls = {'student_network_calls': 0, 'teacher_network_calls': 0, 'cached_field_hits': 0}
    def wrap(field, role):
        @torch.no_grad()
        def predict(x, s):
            s = float(s)
            if s not in parents:
                calls[role+'_network_calls'] += x.shape[0]
                return field(x, s)
            parent = parents[s]
            if not torch.equal(x, parent.expand(x.shape[0], *parent.shape[1:])):
                raise RuntimeError('Last-bank cache saw a changed prefix parent')
            key = (role, s)
            if key not in values:
                values[key] = field(parent, s)
                calls[role+'_network_calls'] += 1
            else:
                calls['cached_field_hits'] += 1
            return values[key].expand(x.shape[0], *values[key].shape[1:])
        return predict
    return wrap(student, 'student'), wrap(teacher, 'teacher'), calls


@torch.no_grad()
def conditional_path_mean(observation, sigma, states, source, student, teacher,
                          positive_sigmas, eta, **kwargs):
    if kwargs.pop('bank_policy', 'last') != 'last':
        raise ValueError('Cached prefix only supports the last innovation bank')
    cached_student, cached_teacher, calls = cached_prefix_fields(student, teacher, states, positive_sigmas)
    mean, metrics = random_walk_mean(observation, sigma, states, source,
        cached_student, cached_teacher, positive_sigmas, eta, bank_policy='last', **kwargs)
    metrics.update({'path_is/'+key: float(value) for key, value in calls.items()})
    metrics['path_is/prefix_cache_exact_checks'] = 1.
    return mean, metrics
