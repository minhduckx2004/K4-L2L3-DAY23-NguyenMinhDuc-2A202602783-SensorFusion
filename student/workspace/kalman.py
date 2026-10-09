"""Extended Kalman filter helpers for 6D constant-velocity motion.

Part E supplies prediction and correction for docs/HUONG_DAN_KY_THUAT.md §2.
Read the shared time step and process-noise settings with get_tracking_params().
"""

from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np
from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: Gợi ý module — params = get_tracking_params() sau khi import ở trên.


def build_F(dt: Optional[float] = None) -> Matrix:
    """Build the constant-velocity state transition matrix F.

    Args:
        dt: Time step in seconds; default from tracking params.

    Returns:
        6x6 state transition matrix as ``np.matrix``.
    """
    params = get_tracking_params()
    step = params.dt if dt is None else float(dt)
    F = np.asmatrix(np.eye(params.dim_state, dtype=float))
    F[0, 3] = step
    F[1, 4] = step
    F[2, 5] = step
    return F


def build_Q(dt: Optional[float] = None, q: Optional[float] = None) -> Matrix:
    """Build the process noise covariance matrix Q.

    Args:
        dt: Time step; default from tracking params.
        q: Process noise scale; default from tracking params.

    Returns:
        6x6 process noise matrix.
    """
    params = get_tracking_params()
    step = params.dt if dt is None else float(dt)
    scale = params.q if q is None else float(q)
    return np.asmatrix(np.eye(params.dim_state, dtype=float) * step * scale)


def ekf_predict(
    x: Matrix,
    P: Matrix,
    F: Optional[Matrix] = None,
    Q: Optional[Matrix] = None,
) -> tuple[Matrix, Matrix]:
    """Predict state and covariance one time step forward.

    Args:
        x: State vector (6x1).
        P: State covariance (6x6).
        F: Optional transition matrix; build via ``build_F`` if None.
        Q: Optional process noise; build via ``build_Q`` if None.

    Returns:
        Tuple ``(x_pred, P_pred)``.
    """
    F_mat = np.asmatrix(build_F() if F is None else F)
    Q_mat = np.asmatrix(build_Q() if Q is None else Q)
    x_mat = np.asmatrix(x)
    P_mat = np.asmatrix(P)
    x_pred = F_mat @ x_mat
    P_pred = F_mat @ P_mat @ F_mat.T + Q_mat
    return np.asmatrix(x_pred), np.asmatrix(P_pred)


def innovation(x: Matrix, meas: Any) -> Matrix:
    """Compute the measurement residual (innovation) gamma.

    Args:
        x: Predicted state.
        meas: Measurement with ``z`` and ``sensor.get_hx(x)``.

    Returns:
        Innovation vector ``z - h(x)``.
    """
    return np.asmatrix(meas.z) - np.asmatrix(meas.sensor.get_hx(x))


def innovation_covariance(P: Matrix, meas: Any, H: Matrix) -> Matrix:
    """Compute the innovation covariance S = H P H' + R.

    Args:
        P: State covariance.
        meas: Measurement with ``R``.
        H: Measurement Jacobian.

    Returns:
        Innovation covariance matrix S.
    """
    H_mat = np.asmatrix(H)
    return H_mat @ np.asmatrix(P) @ H_mat.T + np.asmatrix(meas.R)


def ekf_update(x: Matrix, P: Matrix, meas: Any) -> tuple[Matrix, Matrix]:
    """Apply an EKF measurement update and return updated state and covariance.

    Args:
        x: Prior state.
        P: Prior covariance.
        meas: Associated measurement.

    Returns:
        Tuple ``(x_upd, P_upd)``.
    """
    x_mat = np.asmatrix(x)
    P_mat = np.asmatrix(P)
    H = np.asmatrix(meas.sensor.get_H(x_mat))
    gamma = innovation(x_mat, meas)
    S = innovation_covariance(P_mat, meas, H)
    K = P_mat @ H.T @ np.linalg.inv(S)
    x_upd = x_mat + K @ gamma
    I = np.asmatrix(np.eye(P_mat.shape[0], dtype=float))
    P_upd = (I - K @ H) @ P_mat
    return np.asmatrix(x_upd), np.asmatrix(P_upd)
