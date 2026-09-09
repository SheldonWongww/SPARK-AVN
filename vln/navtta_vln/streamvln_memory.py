"""History selection at a StreamVLN generation-context boundary.

No simulator/model imports: the scheduler and CPU regressions can exercise the
same boundary rule used by the deployed evaluator.
"""


def history_frame_indices(
    starts_new_context, segment_start, num_history, num_future_steps
):
    """Return the historical frames accompanying a fresh memory prompt.

    ``segment_start`` is the first observed step after the last cache reset,
    not necessarily the step at which an unfinished action chunk next empties.
    Preserve the original samples at aligned boundaries. In particular a fresh
    call at step 194 whose segment started at 192 must still carry eight frames.
    """
    if not starts_new_context or segment_start == 0:
        return ()
    if not isinstance(segment_start, int) or segment_start < 0:
        raise ValueError("StreamVLN segment_start must be a nonnegative integer")
    if num_history is None:
        if not isinstance(num_future_steps, int) or num_future_steps < 1:
            raise ValueError("StreamVLN num_future_steps must be positive")
        return tuple(range(0, segment_start, num_future_steps))
    if not isinstance(num_history, int) or not 1 <= num_history <= segment_start:
        raise ValueError("StreamVLN needs num_history frames before the context boundary")
    stride = max(1, segment_start // num_history)
    return tuple(range(0, segment_start, stride))[:num_history]
