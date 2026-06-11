"""Transformation utilities for three-phase inverter signals."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def alpha_beta_transform(data, phase_a, phase_b, phase_c):
    """Apply Clarke transformation (α-β) to three-phase quantities."""
    try:
        sqrt2_3 = np.sqrt(2 / 3)
        transform_matrix = sqrt2_3 * np.array([
            [1.0, -0.5, -0.5],
            [0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2]
        ])
        phases = np.vstack((data[phase_a].to_numpy(),
                            data[phase_b].to_numpy(),
                            data[phase_c].to_numpy())).T
        alpha_beta = phases.dot(transform_matrix.T)
        return alpha_beta[:, 0], alpha_beta[:, 1]
    except KeyError as exc:
        logger.error("Missing column in data: %s", exc)
        raise
    except Exception as exc:
        logger.error("Error in Clarke transformation: %s", exc)
        raise


def park_transform(alpha, beta, theta):
    """Apply Park transformation to convert α-β components to d-q components."""
    try:
        alpha = np.array(alpha)
        beta = np.array(beta)
        theta = np.array(theta)
        if alpha.shape != beta.shape or alpha.shape != theta.shape:
            raise ValueError("Alpha, beta, and theta must have the same length")
        d = np.cos(theta) * alpha + np.sin(theta) * beta
        q = -np.sin(theta) * alpha + np.cos(theta) * beta
        return d, q
    except Exception as exc:
        logger.error("Error in Park transformation: %s", exc)
        raise
