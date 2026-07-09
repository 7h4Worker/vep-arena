"""Signal-level analysis tools for SSVEP experiments.

Provides spectral estimation, SNR computation, phase locking value,
and information-theoretic utilities. Operates on raw or spatially
filtered EEG signals — complementary to the decision-level ``channel``
package which works on confusion matrices.
"""

from vep_arena.signal.spectrum import amplitude_spectrum, compute_psd
from vep_arena.signal.snr import (
    snr_harmonic,
    snr_narrowband,
    snr_wideband,
    ssvep_snr_profile,
)
from vep_arena.signal.plv import itpc, itpc_fft, plv_profile, rayleigh_test
from vep_arena.signal.utils import (
    js_divergence,
    kl_divergence,
    pearson_batch,
    safe_log2,
    spectral_concentration,
)

__all__ = [
    "amplitude_spectrum",
    "compute_psd",
    "itpc",
    "itpc_fft",
    "js_divergence",
    "kl_divergence",
    "pearson_batch",
    "plv_profile",
    "rayleigh_test",
    "safe_log2",
    "snr_harmonic",
    "snr_narrowband",
    "snr_wideband",
    "spectral_concentration",
    "ssvep_snr_profile",
]
