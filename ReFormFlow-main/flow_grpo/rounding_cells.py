"""A posterior-preserving partition for numerically narrow BF16 inverse cells.

Use identity on the unsafe physical-path set A. Every ordinary conditional-iSIR
pool is restricted to A-complement. Both blocks preserve their conditional pi,
so their union preserves the SAME global posterior. An uncorrected reference-
dependent fallback would not have this property. The direct physical last-node
sampler avoids this inverse representation altogether.
"""
import torch


class UnrepresentableRoundingCell(ValueError):
    pass


def unrepresentable_bf16_cells(value, mean, std):
    if value.dtype != torch.bfloat16:
        return torch.zeros_like(value, dtype=torch.bool)
    if std <= 0 or not torch.isfinite(value).all() or not torch.isfinite(mean).all():
        raise ValueError('Nonfinite model state/mean is not an identity-partition case')
    lo = .5*(value.double()+torch.nextafter(value, torch.full_like(value,-float('inf'))).double())
    hi = .5*(value.double()+torch.nextafter(value, torch.full_like(value,float('inf'))).double())
    if not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
        raise ValueError('Overflow rounding cells are unsupported')
    lower,upper = (lo-mean.double())/std,(hi-mean.double())/std
    center = (value.double()-mean.double())/std
    spacing = (torch.nextafter(center,torch.full_like(center,float('inf')))-center).abs()
    return (upper-lower) <= 8*spacing


def identity_partition_result(source, sigma, stop):
    metrics = {f'path_is/{k}':0. for k in ['actual_move_rate','move_rms','mean_displacement_rms',
        'log_weight_range','auxiliary_scale','block_size','path_ratio_span','observation_span',
        'total_mean_displacement_rms','rb_removed_variance_trace_per_coordinate']}
    metrics.update({'path_is/ess':1.,'path_is/max_weight':1.,'path_is/reference_probability':1.,
        'path_is/frozen_update':1.,'path_is/reference_replay_exact':1.,'path_is/sigma':float(sigma),
        'path_is/stop':int(stop),'path_is/identity_due_to_unrepresentable_cell':1.,
        'path_is/candidate_pool_evaluated':0.})
    return source.float(),metrics
