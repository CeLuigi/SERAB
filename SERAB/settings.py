"""Main constants used for the SERAB benchmark."""

# Random seed
RANDOM_SEED = 42

# Required sample rate, in Hz
REQUIRED_SAMPLE_RATE = 16000

# Pre-normalization Mean and SD statistics for each classification dataset
# Format: 'dataset_name': [log_spectrogram_mean_value, log_spectrogram_sd_value]
CLF_STATS_DICT = {
    'cafe': [-7.9731621742248535, 5.112339973449707],
    'emodb': [-2.6927330493927, 4.103071212768555],
    'emovo': [-6.7016167640686035, 4.619407653808594],
    'ravdess': [-10.320347785949707, 4.999551773071289],
    'savee': [-7.362058639526367, 4.717724800109863],
    'shemo': [-3.533191204071045, 4.027871608734131],
    'english': [-6.857725143432617, 4.4416093826293945]
}