def retarget_motion(*args, **kwargs):
    from .pipeline import retarget_motion as run

    return run(*args, **kwargs)


__all__ = ["retarget_motion"]
