"""Main entrypoint for the inverter modeling pipeline."""

import argparse
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from .modeling import build_model, create_pipeline, cross_validate_model, train_model
from .preprocessing import load_and_prepare_data
from .visualization import (
    plot_alpha_beta,
    plot_clarke_park_transformations,
    plot_correlation_heatmap,
    plot_predictions,
    plot_three_phase_waveforms,
    plot_training_history,
)

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def run_pipeline(data_path, sample_size, outlier_threshold, noise_factor):
    configure_logging()
    logger.info('Loading data from %s', data_path)

    features, targets, df, sample_limit = load_and_prepare_data(
        data_path,
        sample_size=sample_size,
        outlier_threshold=outlier_threshold,
        noise_factor=noise_factor,
    )

    plot_three_phase_waveforms(df, sample_size=sample_limit)
    plot_clarke_park_transformations(features, sample_size=sample_limit)
    plot_alpha_beta(features, sample_size=sample_limit)
    plot_correlation_heatmap(features)

    X = features
    y = targets[['d_alpha_k2', 'd_d_k2']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline = create_pipeline()
    pipeline.fit(X_train)
    X_train_scaled = pipeline.transform(X_train)
    X_test_scaled = pipeline.transform(X_test)

    logger.info('Running cross-validation')
    cv_scores = cross_validate_model(build_model, X, y, n_splits=5)
    logger.info('Cross-validation scores: %s', cv_scores)
    logger.info('Mean CV score: %.6f', float(sum(cv_scores) / len(cv_scores)))

    model = build_model(X_train_scaled.shape[1])
    history = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    plot_training_history(history)
    plot_predictions(model, X_test_scaled, y_test)


def main():
    parser = argparse.ArgumentParser(description='Run the inverter modeling pipeline')
    parser.add_argument(
        '--data-path',
        default='data/Inverter Data Set.csv',
        help='Path to the inverter dataset CSV file',
    )
    parser.add_argument('--sample-size', type=int, default=5000, help='Number of rows to use for plotting')
    parser.add_argument('--outlier-threshold', type=float, default=3.0, help='Z-score threshold for outlier capping')
    parser.add_argument('--noise-factor', type=float, default=0.01, help='Noise factor for synthetic data augmentation')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f'Dataset file not found at {data_path}. Place Inverter Data Set.csv in the data/ folder or update --data-path.'
        )

    run_pipeline(
        str(data_path),
        sample_size=args.sample_size,
        outlier_threshold=args.outlier_threshold,
        noise_factor=args.noise_factor,
    )


if __name__ == '__main__':
    main()
