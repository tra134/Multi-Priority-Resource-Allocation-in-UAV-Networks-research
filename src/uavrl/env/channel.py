"""Channel model for the UAV base station.

Implements the Al-Hourani urban air-to-ground (A2G) path-loss model, a sticky
LoS/NLoS Markov chain per traffic class, Rician (LoS) / Rayleigh (NLoS)
small-scale fading with AR(1) correlation across slots, an extra-shadowing
offset hook, and the Shannon rate formula.

All geometry assumes a single UAV hovering at fixed altitude H above users that
lie on the ground at a given radius (fixed per episode).  This is a deliberate
scope limit (plan.md decision D7): there is no trajectory control and no
inter-cell interference.
"""

from __future__ import annotations

import math

import numpy as np

SPEED_OF_LIGHT = 2.99792458e8  # m/s


def p_los(theta_deg, a=9.61, b=0.16):
    """Al-Hourani LoS probability for elevation angle in degrees.

    P_LOS(theta) = 1 / (1 + a * exp(-b * (theta - a)))

    Parameters
    ----------
    theta_deg : float or array-like
        Elevation angle in degrees.
    a, b : float
        Environment parameters (urban: a=9.61, b=0.16).

    Returns
    -------
    float or ndarray
        LoS probability in (0, 1).  Monotone increasing in elevation angle.
    """
    theta = np.asarray(theta_deg, dtype=float)
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


def free_space_path_loss_db(d, fc):
    """Free-space path loss in dB: 20 log10(4 pi d f_c / c)."""
    d = np.asarray(d, dtype=float)
    return 20.0 * np.log10(4.0 * np.pi * d * fc / SPEED_OF_LIGHT)


def noise_power_w(b_hz, noise_dbm_hz, noise_figure_db):
    """Thermal noise power in Watts over bandwidth b_hz.

    N0 [W/Hz] = 10^((noise_dbm_hz + noise_figure_db - 30) / 10).
    """
    n0 = 10.0 ** ((noise_dbm_hz + noise_figure_db - 30.0) / 10.0)
    return n0 * b_hz


def rate(b_hz, p_w, gain, n0_w_per_hz):
    """Shannon rate [bps].

    R = b * log2(1 + p * g / (N0 * b))

    Rate is zero when b = 0 or p = 0.
    """
    b_hz = np.asarray(b_hz, dtype=float)
    p_w = np.asarray(p_w, dtype=float)
    gain = np.asarray(gain, dtype=float)
    snr = np.where(
        (b_hz > 0) & (p_w > 0) & (gain > 0),
        p_w * gain / (n0_w_per_hz * np.maximum(b_hz, 1e-12)),
        0.0,
    )
    return b_hz * np.log2(1.0 + snr)


class Channel:
    """Per-class A2G channel state.

    Each traffic class has an independent sticky LoS/NLoS Markov chain and an
    independent AR(1) fading process.  Users are at a fixed ground radius per
    episode; the UAV hovers at fixed altitude.
    """

    def __init__(
        self,
        n_classes,
        H=100.0,
        fc=2.0e9,
        a=9.61,
        b=0.16,
        eta_los_db=1.0,
        eta_nlos_db=20.0,
        los_stay_prob=0.98,
        rician_K_db=10.0,
        fading_ar1=0.9,
        noise_dbm_hz=-174.0,
        noise_figure_db=7.0,
        rng=None,
    ):
        self.n_classes = n_classes
        self.H = H
        self.fc = fc
        self.a = a
        self.b = b
        self.eta_los_db = eta_los_db
        self.eta_nlos_db = eta_nlos_db
        self.los_stay_prob = los_stay_prob
        self.rician_K_db = rician_K_db
        self.fading_ar1 = fading_ar1
        self.noise_dbm_hz = noise_dbm_hz
        self.noise_figure_db = noise_figure_db
        self.rng = np.random.default_rng() if rng is None else rng

        # Per-class LoS state (bool) and fading state (complex)
        self.los_state = np.zeros(n_classes, dtype=bool)
        self.fading_state = np.ones(n_classes, dtype=complex)

        # Geometry-derived, precomputed once the user radii are known (fixed
        # per episode).  Populated by set_geometry(); update() uses them so the
        # per-slot hot path avoids recomputing distance / elevation / path loss.
        self._d = None
        self._fspl = None
        self._p_los = None
        self._pl_los = None
        self._pl_nlos = None

        # Initialize LoS state from the stationary distribution at broadside
        # (users directly below the UAV, theta = 90 deg) so every class starts
        # in a plausible state.
        p0 = float(p_los(90.0, a, b))
        self.los_state = self.rng.random(n_classes) < p0
        # Large-scale fading mean power = 1; small-scale initialized below.
        self.fading_state = self._draw_fading(self.los_state)

    # ------------------------------------------------------------------ geometry
    def elevation_angle_deg(self, radius_m):
        """Elevation angle from UAV to a user at ground radius [m]."""
        radius_m = np.asarray(radius_m, dtype=float)
        return np.degrees(np.arctan2(self.H, np.maximum(radius_m, 1e-6)))

    def distance_m(self, radius_m):
        """3-D UAV-to-user distance [m]."""
        radius_m = np.asarray(radius_m, dtype=float)
        return np.sqrt(self.H**2 + radius_m**2)

    def set_geometry(self, radius_m):
        """Precompute per-class large-scale geometry (fixed for an episode)."""
        radius_m = np.asarray(radius_m, dtype=float)
        self._d = np.sqrt(self.H**2 + radius_m**2)
        theta = np.degrees(np.arctan2(self.H, np.maximum(radius_m, 1e-6)))
        self._p_los = p_los(theta, self.a, self.b)
        fspl = free_space_path_loss_db(self._d, self.fc)
        self._pl_los = fspl + self.eta_los_db
        self._pl_nlos = fspl + self.eta_nlos_db

    def los_probability(self, radius_m):
        return p_los(self.elevation_angle_deg(radius_m), self.a, self.b)

    def path_loss(self, radius_m, shadow_db=0.0):
        """Mean path loss [dB] including an optional extra-shadowing offset.

        Parameters
        ----------
        radius_m : float or array-like
            User ground radius [m].
        shadow_db : float or array-like
            Extra shadowing offset [dB] added to the path loss (positive =
            deeper fading).  This is the shadowing hook used by scenarios S2/S4/S5.
        """
        if self._d is None or not np.allclose(self._radius if hasattr(self, "_radius") else radius_m, radius_m):
            self.set_geometry(radius_m)
        self._radius = np.asarray(radius_m, dtype=float)
        pl = self._p_los * self._pl_los + (1.0 - self._p_los) * self._pl_nlos
        return pl + np.asarray(shadow_db, dtype=float)

    # ------------------------------------------------------------- small-scale fading
    def _draw_fading(self, los_state):
        """Draw complex fading coefficients per class (vectorized).

        LoS: Rician with K factor (specular + scattered).  NLoS: Rayleigh.
        """
        n = len(los_state)
        K = self.rician_K_linear()
        specular = np.sqrt(K / (K + 1.0))  # unit-power specular component
        scattered_los = np.sqrt(1.0 / (K + 1.0)) / np.sqrt(2.0)
        scattered_nlos = 1.0 / np.sqrt(2.0)
        rx = self.rng.standard_normal(n)
        ry = self.rng.standard_normal(n)
        h_los = specular + scattered_los * (rx + 1j * ry)
        h_nlos = scattered_nlos * (rx + 1j * ry)
        return np.where(los_state, h_los, h_nlos)

    def update(self, shadow_db=0.0):
        """Advance the channel by one slot (Markov LoS state + AR(1) fading).

        Vectorized NumPy over the classes with precomputed geometry; the
        per-slot hot path targets >= 20,000 steps/s on a laptop CPU.  Returns
        the per-class power gain |h|^2 * 10^(-PL/10) including shadowing.
        """
        n = self.n_classes
        sd = float(shadow_db)
        rho = self.fading_ar1
        ar1_term = math.sqrt(max(0.0, 1.0 - rho * rho))

        # Sticky LoS/NLoS Markov chain: stay with prob los_stay_prob, otherwise
        # redraw from the stationary distribution at the current elevation.
        stay = self.rng.random(n) < self.los_stay_prob
        redraw = self.rng.random(n) < self._p_los
        self.los_state = np.where(stay, self.los_state, redraw)

        # AR(1) fading.
        innovation = self._draw_fading(self.los_state)
        self.fading_state = (
            rho * self.fading_state + ar1_term * innovation
        )

        pl = (
            self._p_los * self._pl_los
            + (1.0 - self._p_los) * self._pl_nlos
            + sd
        )
        large_scale = 10.0 ** (-pl / 10.0)
        return np.abs(self.fading_state) ** 2 * large_scale

    def rician_K_linear(self):
        return 10.0 ** (self.rician_K_db / 10.0)

    def gain(self, radius_m, shadow_db=0.0):
        """Instantaneous per-class power gain |h|^2 * 10^(-PL/10) without advancing state."""
        if self._d is None:
            self.set_geometry(radius_m)
        pl = (
            self._p_los * self._pl_los
            + (1.0 - self._p_los) * self._pl_nlos
            + np.asarray(shadow_db, dtype=float)
        )
        return np.abs(self.fading_state) ** 2 * 10.0 ** (-pl / 10.0)

    # --------------------------------------------------------------------- rate
    def rate_bps(self, b_hz, p_w, gain):
        """Shannon rate [bps] for given bandwidth [Hz], power [W], gain."""
        return rate(b_hz, p_w, gain, noise_power_w(1.0, self.noise_dbm_hz, self.noise_figure_db))
