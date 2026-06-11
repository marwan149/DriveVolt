"""Data preparation and feature engineering for the inverter dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .transforms import alpha_beta_transform, park_transform

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'u_a_k-1', 'u_b_k-1', 'u_c_k-1',
    'i_a_k-3', 'i_b_k-3', 'i_c_k-3',
    'i_a_k-2', 'i_b_k-2', 'i_c_k-2',
    'd_a_k-3', 'd_b_k-3', 'd_c_k-3',
    'd_a_k-2', 'd_b_k-2', 'd_c_k-2',
    'u_dc_k-3', 'u_dc_k-2'
]
OPTIONAL_SPEED_COLUMN = 'n_k'


def estimate_theta(df, time_step=1e-4, default_freq=50):
    if OPTIONAL_SPEED_COLUMN in df.columns:
        return np.cumsum(df[OPTIONAL_SPEED_COLUMN].to_numpy() * time_step * 2 * np.pi / 60)

    logger.warning("Speed column '%s' not found. Assuming %s Hz for theta.", OPTIONAL_SPEED_COLUMN, default_freq)
    t = np.arange(len(df)) * time_step
    return 2 * np.pi * default_freq * t


def load_and_prepare_data(file_path, sample_size=5000, outlier_threshold=3, noise_factor=0.01):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        logger.warning("Missing values detected. Filling missing values with median.")
        df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].fillna(df[REQUIRED_COLUMNS].median())

    for col in REQUIRED_COLUMNS:
        column_std = df[col].std()
        if column_std == 0 or np.isnan(column_std):
            continue
        z_scores = np.abs((df[col] - df[col].mean()) / column_std)
        df[col] = np.where(
            z_scores > outlier_threshold,
            np.sign(df[col]) * column_std * outlier_threshold + df[col].mean(),
            df[col]
        )

    u_alpha, u_beta = alpha_beta_transform(df, 'u_a_k-1', 'u_b_k-1', 'u_c_k-1')
    i_alpha_k3, i_beta_k3 = alpha_beta_transform(df, 'i_a_k-3', 'i_b_k-3', 'i_c_k-3')
    i_alpha_k2, i_beta_k2 = alpha_beta_transform(df, 'i_a_k-2', 'i_b_k-2', 'i_c_k-2')
    d_alpha_k3, d_beta_k3 = alpha_beta_transform(df, 'd_a_k-3', 'd_b_k-3', 'd_c_k-3')
    d_alpha_k2, d_beta_k2 = alpha_beta_transform(df, 'd_a_k-2', 'd_b_k-2', 'd_c_k-2')

    theta = estimate_theta(df)

    u_d, u_q = park_transform(u_alpha, u_beta, theta)
    i_d_k3, i_q_k3 = park_transform(i_alpha_k3, i_beta_k3, theta)
    i_d_k2, i_q_k2 = park_transform(i_alpha_k2, i_beta_k2, theta)
    d_d_k3, d_q_k3 = park_transform(d_alpha_k3, d_beta_k3, theta)
    d_d_k2, d_q_k2 = park_transform(d_alpha_k2, d_beta_k2, theta)

    features = pd.DataFrame({
        'u_alpha': u_alpha,
        'u_beta': u_beta,
        'u_d': u_d,
        'u_q': u_q,
        'd_alpha_k3': d_alpha_k3,
        'd_beta_k3': d_beta_k3,
        'd_d_k3': d_d_k3,
        'd_q_k3': d_q_k3,
        'i_alpha_k3': i_alpha_k3,
        'i_beta_k3': i_beta_k3,
        'i_d_k3': i_d_k3,
        'i_q_k3': i_q_k3,
        'i_alpha_k2': i_alpha_k2,
        'i_beta_k2': i_beta_k2,
        'i_d_k2': i_d_k2,
        'i_q_k2': i_q_k2,
        'u_dc_k3': df['u_dc_k-3'].to_numpy(),
        'u_dc_k2': df['u_dc_k-2'].to_numpy(),
        'i_alpha_diff': i_alpha_k2 - i_alpha_k3,
        'i_beta_diff': i_beta_k2 - i_beta_k3,
        'i_d_diff': i_d_k2 - i_d_k3,
        'i_q_diff': i_q_k2 - i_q_k3,
        'u_dc_diff': df['u_dc_k-2'].to_numpy() - df['u_dc_k-3'].to_numpy()
    })

    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    if to_drop:
        logger.info("Dropping highly correlated features: %s", to_drop)
        features = features.drop(columns=to_drop)

    noise = np.random.normal(0, 1.0, size=features.shape) * features.std().to_numpy()
    noisy_features = features + noise
    features = pd.concat([features, noisy_features], ignore_index=True)

    targets = pd.DataFrame({
        'd_alpha_k2': np.concatenate([d_alpha_k2, d_alpha_k2]),
        'd_beta_k2': np.concatenate([d_beta_k2, d_beta_k2]),
        'd_d_k2': np.concatenate([d_d_k2, d_d_k2]),
        'd_q_k2': np.concatenate([d_q_k2, d_q_k2])
    }, index=features.index)

    return features, targets, df, min(sample_size, len(df))
