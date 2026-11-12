"""Supervised foreground correction with explicit, user-supplied references.

Models are JSON-compatible dictionaries. Training never constructs a reference
from simulator truth or silently aligns predictions to test data. Features may
be final codes, stage subcodes, or histories, but their meaning and units must
stay the same between fitting and application.

The learned coefficients are digital correction parameters, not identified ADC
capacitor weights or amplifier gains. Normal operation evaluates the frozen map
after the ADC; it needs no reference and performs no parameter adaptation.
"""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize


def _features(values, *, empty=False):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] == 0 or (not empty and len(x) == 0) or not np.isfinite(x).all():
        raise ValueError("features must be a finite (samples, features) matrix or 1D signal")
    return x


def _unpack_features(features, *, empty=False):
    """Return the feature matrix and, when known, its channel/history layout.

    ``features`` is either a matrix or the dict returned by
    ``build_calibration_features``. Only the dict carries the layout.
    """
    layout = None
    if isinstance(features, dict):
        if "features" not in features:
            raise ValueError('a feature dict must contain "features"')
        layout = features.get("layout")
        if layout is not None:
            layout = {"channels": int(layout["channels"]), "history": int(layout["history"])}
        features = features["features"]
    return _features(features, empty=empty), layout


def _check_layout(expected, actual, what):
    if expected is not None and actual is not None and expected != actual:
        raise ValueError(
            f"feature layout mismatch: {what} uses channels={actual['channels']}, "
            f"history={actual['history']}, but the model was built for "
            f"channels={expected['channels']}, history={expected['history']}"
        )


def _reference(values, size):
    y = np.asarray(values, dtype=float)
    if y.shape != (size,) or not np.isfinite(y).all():
        raise ValueError("reference must be a finite 1D array matching the sample count")
    return y


def build_calibration_features(samples, *, history=0, state=None):
    """Build current-first delayed features without fabricating startup samples.

    Parameters
    ----------
    samples : array_like
        Shape (N,) or (N, C), in a fixed channel order and scale.
    history : int
        Number of preceding samples required in each feature row.
    state : array_like or None
        Previous returned ``state`` for a contiguous stream. Pass None at the
        start of every independent record; never connect unrelated waveforms.

    Returns
    -------
    dict
        ``features`` has columns [current channels, lag-1 channels, ...].
        ``sample_indices`` selects matching labels from this input block.
        ``state`` contains at most history final input rows, oldest first.
        ``layout`` records the channel count and history. Pass this dict
        itself (not only ``features``) to the fit functions and to
        ``apply_calibration`` to have the layout stored in the model and
        checked at application: two layouts with the same number of columns
        (two channels with history 1, one channel with history 3) are
        otherwise indistinguishable.
        A short startup block may return zero feature rows.
    """
    if isinstance(history, bool) or not isinstance(history, (int, np.integer)) or history < 0:
        raise ValueError("history must be a nonnegative integer")
    x = _features(samples)
    previous = np.empty((0, x.shape[1])) if state is None else _features(state, empty=True)
    if previous.shape[1] != x.shape[1] or len(previous) > history:
        raise ValueError("state must have the same channel count and at most history rows")
    joined = np.vstack([previous, x])
    indices = np.arange(max(0, history-len(previous)), len(x), dtype=int)
    pos = indices + len(previous)
    features = np.concatenate([joined[pos-lag] for lag in range(history+1)], axis=1)
    tail = joined[-history:].copy() if history else np.empty((0, x.shape[1]))
    return {"features": features, "sample_indices": indices, "state": tail,
            "layout": {"channels": int(x.shape[1]), "history": int(history)}}


def _terms(count, degree):
    # Full total-degree polynomial, including cross terms and one intercept.
    return [()] + [t for d in range(1, degree+1)
                   for t in combinations_with_replacement(range(count), d)]


def _design(z, terms):
    return np.column_stack([np.prod(z[:, t], axis=1) if t else np.ones(len(z)) for t in terms])


def _base(x):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < np.finfo(float).eps * np.maximum(1, abs(mean))] = 1.
    model = {"schema_version": 1, "feature_count": x.shape[1],
             "input_mean": mean.tolist(), "input_scale": scale.tolist(),
             "training_min": x.min(axis=0).tolist(), "training_max": x.max(axis=0).tolist()}
    return model, (x-mean)/scale


def _ridge(design, y, ridge):
    penalty = np.eye(design.shape[1]) * np.sqrt(ridge)
    penalty[0, 0] = 0.  # Do not penalize the intercept.
    augmented = np.vstack([design/np.sqrt(len(y)), penalty])
    target = np.r_[y/np.sqrt(len(y)), np.zeros(len(penalty))]
    coefficients = np.linalg.lstsq(augmented, target, rcond=None)[0]
    return coefficients


def fit_calibration_polynomial(features, reference, *, degree=3, ridge=1e-8):
    """Fit a direct inverse correction to an aligned, supplied reference.

    All monomials of total degree <= degree are included. Features are scaled
    using training data only. ``ridge`` penalizes coefficients in that scaled
    basis: mean squared reference error + ridge * sum(non-intercept weights**2).
    The mapping is nonlinear in inputs for degree > 1, although fitting is
    linear in coefficients. More than 4096 terms is rejected to bound memory.

    Returns a dict with a JSON-compatible ``model`` and fitting ``diagnostics``.
    Apply the model to independent data using :func:`apply_calibration`.
    No extrapolation accuracy, monotonicity or reference correctness is implied.
    ``design_condition_number`` diagnoses the standardized, unregularized design;
    it is None when numerically rank deficient, including underdetermined fits.
    Full rank and a finite condition number do not establish joint coverage.
    """
    from math import comb
    x, layout = _unpack_features(features)
    y = _reference(reference, len(x))
    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if comb(x.shape[1]+degree, degree) > 4096:
        raise ValueError("polynomial exceeds 4096 terms; reduce features or degree")
    model, z = _base(x)
    terms = _terms(x.shape[1], degree)
    design = _design(z, terms)
    coefficients = _ridge(design, y, ridge)
    singular = np.linalg.svd(design, compute_uv=False)
    threshold = np.finfo(float).eps * max(design.shape) * singular[0]
    rank = int(np.sum(singular > threshold))
    condition = float(singular[0]/singular[-1]) if rank == len(terms) else None
    model.update(kind="polynomial", degree=int(degree), coefficients=coefficients.tolist())
    if layout is not None:
        model["feature_layout"] = layout
    return {"model": model, "diagnostics": {"samples": len(x), "parameters": len(terms),
            "design_rank": rank, "rank_deficient": rank < len(terms), "ridge": float(ridge),
            "design_condition_number": condition,
            "training_rmse": float(np.sqrt(np.mean((design@coefficients-y)**2)))}}


def fit_calibration_mlp(features, reference, *, validation_features, validation_reference,
                        hidden_units=16, max_iterations=200, l2=1e-6, seed=1):
    """Fit a small tanh residual MLP, selecting an iterate on held-out data.

    A training-only affine baseline is frozen; the fully trained one-hidden-layer
    network learns its residual. L-BFGS uses full-batch squared error and L2 weight
    regularization. Normalization and residual scale are fixed from training data.
    The selected model minimizes validation reference MSE over the initial state
    and optimizer iterates. Validation must be separate records, not test data.
    Returns ``model``, ``diagnostics`` and per-iterate ``history``. NumPy and SciPy
    suffice; no neural-network dependency is required. This is foreground fitting,
    not online adaptation. ``max_iterations`` is an optimizer budget, not epochs.
    The network estimates a residual correction to the affine baseline; its
    weights are not physical ADC parameters. See the packaged FOREGROUND_THEORY.md
    for equations, reference requirements and measured-hardware literature.
    """
    x, layout = _unpack_features(features)
    y = _reference(reference, len(x))
    xv, validation_layout = _unpack_features(validation_features)
    yv = _reference(validation_reference, len(xv))
    if xv.shape[1] != x.shape[1]:
        raise ValueError("training and validation feature counts must match")
    _check_layout(layout, validation_layout, "validation_features")
    for name, value in (("hidden_units", hidden_units), ("max_iterations", max_iterations)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    model, z = _base(x)
    zv = (xv-np.asarray(model["input_mean"]))/np.asarray(model["input_scale"])
    affine = _ridge(np.column_stack([np.ones(len(z)), z]), y, 1e-10)
    residual = y - affine[0] - z@affine[1:]
    scale = max(float(np.std(residual)), 1e-12)
    target = residual/scale
    target_v = (yv-affine[0]-zv@affine[1:])/scale
    count, hidden = x.shape[1], int(hidden_units)
    # Resolve the seed before the fit: an unusable value used to fail only
    # after the whole optimization had run.
    reported_seed = None if seed is None else int(seed)
    rng = np.random.default_rng(reported_seed)
    # Nonzero output weights allow every layer to learn from the first step.
    initial = np.r_[rng.normal(0, .5/np.sqrt(count), count*hidden),
                    rng.normal(0, .15, hidden), rng.normal(0, .01, hidden), 0.]

    def unpack(theta):
        end = count*hidden
        return theta[:end].reshape(count, hidden), theta[end:end+hidden], theta[end+hidden:-1], theta[-1]

    def objective(theta):
        w, b, v, c = unpack(theta)
        h = np.tanh(z@w+b)
        error = h@v+c-target
        derivative = 2*error/len(error)
        dh = derivative[:, None]*v*(1-h*h)
        gradient = np.r_[(z.T@dh+2*l2*w).ravel(), dh.sum(axis=0),
                         h.T@derivative+2*l2*v, derivative.sum()]
        loss = np.mean(error**2)+l2*(np.sum(w*w)+np.sum(v*v))
        return float(loss), gradient

    history = []
    last_observed = None
    best = {"loss": np.inf, "theta": initial.copy(), "iteration": 0}

    def observe(theta):
        nonlocal last_observed
        w, b, v, c = unpack(theta)
        validation_mse = float(np.mean((np.tanh(zv@w+b)@v+c-target_v)**2)*scale**2)
        training_mse = float(np.mean((np.tanh(z@w+b)@v+c-target)**2)*scale**2)
        history.append({"iteration": len(history), "training_mse": training_mse,
                        "validation_mse": validation_mse})
        last_observed = theta.copy()
        if validation_mse < best["loss"]:
            best.update(loss=validation_mse, theta=theta.copy(), iteration=len(history)-1)

    observe(initial)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", callback=observe,
                      options={"maxiter": int(max_iterations), "ftol": 1e-12, "gtol": 1e-8, "maxls": 40})
    if not np.array_equal(result.x, last_observed):
        # Record a final point only if the callback has not already observed it.
        observe(result.x)
    w, b, v, c = unpack(best["theta"])
    if layout is not None:
        model["feature_layout"] = layout
    model.update(kind="mlp", affine=affine.tolist(), residual_scale=scale,
                 hidden_weights=w.tolist(), hidden_bias=b.tolist(),
                 output_weights=v.tolist(), output_bias=float(c))
    return {"model": model, "history": history, "diagnostics": {
        "samples": len(x), "validation_samples": len(xv),
        "parameters": len(initial)+len(affine), "selected_iteration": best["iteration"],
        "iterations": int(result.nit), "optimizer_converged": bool(result.success),
        "optimizer_message": str(result.message), "validation_rmse": float(np.sqrt(best["loss"])),
        "seed": reported_seed, "l2": float(l2)}}


def apply_calibration(features, model):
    """Apply a frozen polynomial/MLP model without refitting or changing state.

    Returns ``calibrated_signal`` and a per-row ``outside_training_range`` mask.
    The mask checks marginal feature ranges, not joint coverage or reliability.
    Extrapolated values are returned without clipping. Models are ordinary
    JSON-compatible dictionaries produced by the fit functions.

    ``features`` may be a matrix or the dict from
    ``build_calibration_features``. With the dict, a model that recorded its
    feature layout rejects features built with a different channel count or
    history; a bare matrix is only checked for its number of columns.
    """
    x, layout = _unpack_features(features, empty=True)
    if model.get("schema_version") != 1 or model.get("feature_count") != x.shape[1]:
        raise ValueError("unsupported model schema or mismatched feature count")
    _check_layout(model.get("feature_layout"), layout, "features")
    mean, scale, low, high = [np.asarray(model[k], float) for k in
                              ("input_mean", "input_scale", "training_min", "training_max")]
    if any(a.shape != (x.shape[1],) or not np.isfinite(a).all() for a in (mean, scale, low, high)) or np.any(scale <= 0) or np.any(low > high):
        raise ValueError("invalid model normalization or training range")
    z = (x-mean)/scale
    if model.get("kind") == "polynomial":
        degree = model["degree"]
        from math import comb
        if not isinstance(degree, int) or degree < 1 or comb(x.shape[1]+degree, degree) > 4096:
            raise ValueError("invalid polynomial degree or size")
        terms = _terms(x.shape[1], degree)
        coefficients = np.asarray(model["coefficients"], float)
        if coefficients.shape != (len(terms),) or not np.isfinite(coefficients).all():
            raise ValueError("invalid polynomial coefficients")
        prediction = _design(z, terms)@coefficients
    elif model.get("kind") == "mlp":
        affine, w, b, v = [np.asarray(model[k], float) for k in
                           ("affine", "hidden_weights", "hidden_bias", "output_weights")]
        s, c = model["residual_scale"], model["output_bias"]
        if (w.ndim != 2 or w.shape[0] != x.shape[1] or w.shape[1] < 1
                or b.shape != (w.shape[1],) or v.shape != b.shape or affine.shape != (x.shape[1]+1,)
                or any(not a.size or not np.isfinite(a).all() for a in (affine, w, b, v))
                or not np.isfinite([s, c]).all() or s <= 0):
            raise ValueError("invalid MLP parameters")
        prediction = affine[0]+z@affine[1:]+s*(np.tanh(z@w+b)@v+c)
    else:
        raise ValueError("unknown calibration model kind")
    if not np.isfinite(prediction).all():
        raise ValueError("calibrated output is nonfinite; check input range and model")
    return {"calibrated_signal": prediction,
            "outside_training_range": np.any((x < low) | (x > high), axis=1)}
