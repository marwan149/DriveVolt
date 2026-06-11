"""Visualization utilities for inverter data and model results."""

import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_three_phase_waveforms(df, sample_size=1000):
    df_sample = df.head(sample_size)
    phase_columns = {
        'currents': [col for col in df_sample.columns if col.startswith('i_') and col.endswith('-2')],
        'voltages': [col for col in df_sample.columns if col.startswith('u_') and col.endswith('-1') or col.endswith('-2')],
        'dc': [col for col in df_sample.columns if col.startswith('u_dc_')],
        'speed': ['n_k']
    }

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), constrained_layout=True)

    if phase_columns['currents']:
        for column in sorted(phase_columns['currents']):
            axes[0].plot(df_sample[column], label=column)
        axes[0].set_title('Phase Currents')
        axes[0].set_ylabel('Current')
        axes[0].legend()
        axes[0].grid(True)
    else:
        logger.warning('No current columns found for plotting')

    if phase_columns['voltages']:
        for column in sorted(phase_columns['voltages']):
            axes[1].plot(df_sample[column], label=column)
        axes[1].set_title('Phase Voltages')
        axes[1].set_ylabel('Voltage')
        axes[1].legend()
        axes[1].grid(True)
    else:
        logger.warning('No voltage columns found for plotting')

    if phase_columns['dc']:
        for column in sorted(phase_columns['dc']):
            axes[2].plot(df_sample[column], label=column)
        axes[2].set_title('DC-link Voltage')
        axes[2].set_ylabel('Voltage')
        axes[2].legend()
        axes[2].grid(True)
    else:
        logger.warning('No DC-link voltage columns found for plotting')

    if phase_columns['speed'][0] in df_sample.columns:
        axes[3].plot(df_sample[phase_columns['speed'][0]], label='n_k')
        axes[3].set_title('Speed')
        axes[3].set_ylabel('Speed')
        axes[3].set_xlabel('Sample')
        axes[3].grid(True)
    else:
        axes[3].text(0.5, 0.5, 'Speed data not available', ha='center', va='center')
        axes[3].set_axis_off()
        logger.warning('Speed column not found for plotting')

    return fig


def plot_clarke_park_transformations(features, sample_size=5000):
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), constrained_layout=True)
    axes = axes.ravel()

    mapping = [
        ('u_alpha', 'u_beta', 'Voltages (α-β)'),
        ('u_d', 'u_q', 'Voltages (d-q)'),
        ('i_alpha_k3', 'i_beta_k3', 'Currents k-3 (α-β)'),
        ('i_d_k3', 'i_q_k3', 'Currents k-3 (d-q)'),
        ('i_alpha_k2', 'i_beta_k2', 'Currents k-2 (α-β)'),
        ('i_d_k2', 'i_q_k2', 'Currents k-2 (d-q)'),
        ('d_alpha_k3', 'd_beta_k3', 'Duties k-3 (α-β)'),
        ('d_d_k3', 'd_q_k3', 'Duties k-3 (d-q)'),
        ('d_alpha_k2', 'd_beta_k2', 'Duties k-2 (α-β)'),
        ('d_d_k2', 'd_q_k2', 'Duties k-2 (d-q)')
    ]

    for ax, (x_col, y_col, title) in zip(axes, mapping):
        if x_col in features.columns and y_col in features.columns:
            ax.plot(features[x_col].head(sample_size), features[y_col].head(sample_size), '.', alpha=0.5)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Clarke/Park {title}')
            ax.grid(True)
        else:
            ax.axis('off')
            logger.warning('Skipping plot %s vs %s because columns are missing', x_col, y_col)

    return fig


def plot_alpha_beta(features, sample_size=1000):
    fig, axes = plt.subplots(3, 2, figsize=(16, 16), constrained_layout=True)
    axes = axes.ravel()

    def safe_plot(ax, cols, title, ylabel):
        available = [col for col in cols if col in features.columns]
        if available:
            for col in available:
                ax.plot(features[col].head(sample_size), label=col)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)
        else:
            ax.axis('off')
            logger.warning('Skipping %s because no columns are available', title)

    safe_plot(axes[0], ['u_alpha', 'u_beta'], 'Alpha-Beta Voltages', 'Voltage')
    safe_plot(axes[1], ['u_d', 'u_q'], 'd-q Voltages', 'Voltage')
    safe_plot(axes[2], ['i_alpha_k2', 'i_beta_k2'], 'Alpha-Beta Currents k-2', 'Current')
    safe_plot(axes[3], ['i_d_k2', 'i_q_k2'], 'd-q Currents k-2', 'Current')
    safe_plot(axes[4], ['d_alpha_k3', 'd_beta_k3'], 'Alpha-Beta Duties k-3', 'Duty')
    safe_plot(axes[5], ['d_d_k3', 'd_q_k3'], 'd-q Duties k-3', 'Duty')

    return fig


def plot_correlation_heatmap(features):
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
    sns.heatmap(features.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    return fig


def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    return fig


def plot_predictions(model, X_test, y_test, sample_size=200):
    predictions = model.predict(X_test[:sample_size], verbose=0)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    if 'd_alpha_k2' in y_test.columns:
        axes[0].plot(y_test['d_alpha_k2'].head(sample_size).values, label='Actual α')
        axes[0].plot(predictions[:, 0], '--', label='Predicted α')
        axes[0].set_title('Actual vs Predicted d_alpha_k2')
        axes[0].legend()
        axes[0].grid(True)
    else:
        logger.warning('d_alpha_k2 not found in y_test')
        axes[0].axis('off')

    if 'd_d_k2' in y_test.columns:
        axes[1].plot(y_test['d_d_k2'].head(sample_size).values, label='Actual d')
        axes[1].plot(predictions[:, 1], '--', label='Predicted d')
        axes[1].set_title('Actual vs Predicted d_d_k2')
        axes[1].legend()
        axes[1].grid(True)
    else:
        logger.warning('d_d_k2 not found in y_test')
        axes[1].axis('off')

    return fig
