"""A receiver that moves: its own Doppler is part of solve_multinode's model."""

import math

import numpy as np
import pytest

from retina_geolocator import Geometry
from retina_geolocator.multinode_solver import (
    _SIGMA_DELAY_US,
    _SIGMA_DOPPLER_HZ,
    MultiNodeMeasurement,
    NodeSetup,
    _jacobian_function,
    _jacobian_function_free,
    _lla_to_enu_km,
    _residual_function,
    _residual_function_free,
    _rx_velocity_in_frame_kms,
    solve_multinode,
)

C = 299_792_458.0
FC = 600e6
RX = (49.26, -123.25, 50.0)
TXS = {"a": (49.40, -123.00, 300.0), "b": (49.10, -123.05, 250.0), "c": (49.35, -123.55, 400.0)}
RX_VEL = (12.0, -7.0, 0.5)
TARGET = (49.33, -123.12, 6000.0)
TARGET_VEL = (180.0, -60.0, 0.0)


def _ecef(lat, lon, alt_m):
    return np.array(Geometry.lla2ecef(lat, lon, alt_m))


def _enu_vec_to_ecef(v, lat, lon, alt_m):
    return np.array(Geometry.enu2ecef(*v, lat, lon, alt_m)) - _ecef(lat, lon, alt_m)


def _bistatic_range_m(p, rx, tx):
    return np.linalg.norm(p - tx) + np.linalg.norm(p - rx) - np.linalg.norm(rx - tx)


def _measure(rx_vel=RX_VEL):
    """Delay and Doppler per node from first principles: the bistatic range and
    -(1/lambda) d(range)/dt by central difference, target and receiver both moving."""
    p0, rx0 = _ecef(*TARGET), _ecef(*RX)
    vp = _enu_vec_to_ecef(TARGET_VEL, *TARGET)
    vr = _enu_vec_to_ecef(rx_vel, *RX)
    out = []
    for nid, tx in TXS.items():
        txe = _ecef(*tx)
        h = 1e-3
        r_plus = _bistatic_range_m(p0 + vp * h, rx0 + vr * h, txe)
        r_minus = _bistatic_range_m(p0 - vp * h, rx0 - vr * h, txe)
        doppler = -(FC / C) * (r_plus - r_minus) / (2 * h)
        out.append({"node_id": nid, "delay_us": _bistatic_range_m(p0, rx0, txe) / C * 1e6, "doppler_hz": doppler})
    return out


def _configs(rx_vel=RX_VEL):
    cfgs = {}
    for nid, tx in TXS.items():
        cfg = {"rx_lat": RX[0], "rx_lon": RX[1], "rx_alt_ft": RX[2] / 0.3048, "tx_lat": tx[0], "tx_lon": tx[1],
               "tx_alt_ft": tx[2] / 0.3048, "fc_hz": FC}  # fmt: skip
        if rx_vel is not None:
            cfg.update(rx_vel_east_ms=rx_vel[0], rx_vel_north_ms=rx_vel[1], rx_vel_up_ms=rx_vel[2])
        cfgs[nid] = cfg
    return cfgs


def _input(meas, alt_km=TARGET[2] / 1000):
    return {"initial_guess": {"lat": TARGET[0] + 0.02, "lon": TARGET[1] - 0.02, "alt_km": alt_km},
            "measurements": meas, "n_nodes": len(meas), "timestamp_ms": 1}  # fmt: skip


def _setups(ref):
    setups = {}
    for nid, tx in TXS.items():
        rx = _lla_to_enu_km(*RX, *ref)
        txe = _lla_to_enu_km(*tx, *ref)
        setups[nid] = NodeSetup(nid, rx, txe, FC, _rx_velocity_in_frame_kms(RX_VEL, *RX, *ref))
    return setups


def test_model_matches_first_principles_doppler():
    ref = (TARGET[0] + 0.05, TARGET[1] + 0.05, 0.0)
    setups = _setups(ref)
    p = _lla_to_enu_km(*TARGET, *ref)
    v = np.array(_rx_velocity_in_frame_kms(TARGET_VEL, *TARGET, *ref)) * 1e3
    meas = [MultiNodeMeasurement(m["node_id"], m["delay_us"], m["doppler_hz"]) for m in _measure()]
    r = _residual_function(np.array([p[0], p[1], *v]), setups, meas, p[2])
    assert np.abs(r[0::2]).max() * _SIGMA_DELAY_US < 1e-4
    assert np.abs(r[1::2]).max() * _SIGMA_DOPPLER_HZ < 0.01
    without = {nid: NodeSetup(nid, s.rx_enu, s.tx_enu, FC) for nid, s in setups.items()}
    r0 = _residual_function(np.array([p[0], p[1], *v]), without, meas, p[2])
    assert np.abs(r0[1::2]).max() * _SIGMA_DOPPLER_HZ > 5.0


@pytest.mark.parametrize("free", [False, True])
def test_analytic_jacobian_matches_finite_difference(free):
    ref = (TARGET[0] + 0.05, TARGET[1] + 0.05, 0.0)
    setups = _setups(ref)
    meas = [MultiNodeMeasurement(m["node_id"], m["delay_us"], m["doppler_hz"], snr=15) for m in _measure()]
    if free:
        state = np.array([3.0, -2.0, 6.5, 150.0, -40.0, 2.0])
        fn, jac, args = _residual_function_free, _jacobian_function_free, (setups, meas)
    else:
        state = np.array([3.0, -2.0, 150.0, -40.0, 2.0])
        fn, jac, args = _residual_function, _jacobian_function, (setups, meas, 6.5)
    J = jac(state, *args)
    num = np.zeros_like(J)
    for k in range(len(state)):
        h = 1e-6 * max(1.0, abs(state[k]))
        d = np.zeros_like(state)
        d[k] = h
        num[:, k] = (fn(state + d, *args) - fn(state - d, *args)) / (2 * h)
    assert np.allclose(J, num, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("free", [False, True])
def test_noise_free_solve_recovers_target_with_moving_receiver(free):
    res = solve_multinode(_input(_measure()), _configs(), free_altitude=free)
    assert res and res["success"]
    north = math.radians(res["lat"] - TARGET[0]) * 6371e3
    east = math.radians(res["lon"] - TARGET[1]) * 6371e3 * math.cos(math.radians(TARGET[0]))
    assert math.hypot(east, north) < 5.0
    assert math.hypot(res["vel_east"] - TARGET_VEL[0], res["vel_north"] - TARGET_VEL[1]) < 0.5
    assert res["rms_doppler"] < 0.05


def test_ignoring_receiver_velocity_biases_aircraft_velocity():
    res = solve_multinode(_input(_measure()), _configs(rx_vel=None))
    assert math.hypot(res["vel_east"] - TARGET_VEL[0], res["vel_north"] - TARGET_VEL[1]) > 5.0


@pytest.mark.parametrize("vel", [None, (0.0, 0.0, 0.0)])
def test_stationary_receiver_is_solved_exactly_as_before(vel):
    meas = _measure(rx_vel=(0.0, 0.0, 0.0))
    base = solve_multinode(_input(meas), _configs(rx_vel=None))
    cfgs = _configs(rx_vel=vel)
    for cfg in cfgs.values():
        cfg.setdefault("rx_vel_east_ms", None)
    assert solve_multinode(_input(meas), cfgs) == base
