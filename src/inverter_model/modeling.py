"""Modeling utilities for inverter data training and evaluation."""

import logging
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models

logger = logging.getLogger(__name__)


def create_pipeline():
    return Pipeline([
        ('scaler', StandardScaler())
    ])


def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    return history


def cross_validate_model(build_model_fn, X, y, n_splits=5, epochs=30, batch_size=256):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        logger.info('Starting fold %d/%d', fold, n_splits)
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = create_pipeline()
        pipeline.fit(X_train)
        X_train_scaled = pipeline.transform(X_train)
        X_val_scaled = pipeline.transform(X_val)

        model = build_model_fn(X_train_scaled.shape[1])
        history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=epochs, batch_size=batch_size)
        fold_scores.append(min(history.history['val_loss']))
        logger.info('Fold %d best val_loss: %.6f', fold, fold_scores[-1])

    return fold_scores
