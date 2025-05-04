from crism_ml.preprocessing import BANDS, N_BANDS, _resample_convhull
import numpy as np


def remove_continuum(signals, bands=None):
    """Remove the slowly-varying component from multiple spectra efficiently.

    Parameters
    ----------
    signals: ndarray
        2D array of signals to remove the continuum from (shape: n_samples x n_features)
    bands: ndarray
        signal bands; defaults to the 248 main bands

    Returns
    -------
    flat_sigs: ndarray
        signals without the continuum (same shape as input)
    curves: ndarray
        the continuum curves (same shape as input)
    """
    if bands is None:
        bands = BANDS[:N_BANDS]

    signals = np.atleast_2d(signals)
    not_const = np.ptp(signals, axis=1) > 0
    flat_sigs = np.zeros_like(signals)
    
    # Pre-allocate curves array
    curves = np.zeros_like(signals)
    
    # Process only non-constant signals
    if np.any(not_const):
        # Vectorized processing of non-constant signals
        curves[not_const] = np.array([_resample_convhull(s, bands) for s in signals[not_const]])
        with np.errstate(invalid='ignore'):
            flat_sigs[not_const] = signals[not_const] / curves[not_const]

    return flat_sigs, curves


