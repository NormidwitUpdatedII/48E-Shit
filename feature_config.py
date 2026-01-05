# Feature Engineering Configuration for Inflation Forecasting
# Configuration parameters for advanced feature transformations

# Raw feature indices (V1-V127 for full FRED-MD, but we use available features)
# In our rawdata.csv, we have the columns already transformed

# Rolling Window Configurations
ROLLING_WINDOWS = [3, 6, 12]

# Momentum Configuration  
MOMENTUM_HORIZONS = [3, 6, 12]

# Volatility Windows
VOLATILITY_WINDOWS = [3, 6, 12]

# Z-score Configuration
ZSCORE_WINDOW = 12
OUTLIER_THRESHOLD = 2.0

# PCA Configuration
N_PCA_COMPONENTS = 4

# Target columns (1-indexed as in R)
CPI_INDEX = 1  # Column index for CPI inflation
PCE_INDEX = 2  # Column index for PCE inflation

# Embedding dimension for lagged features
EMBED_DIMENSION = 4

# Cross-sectional feature pairs (if applicable)
# These are indices in rawdata.csv that represent key economic indicators
CROSS_SECTIONAL_PAIRS = {
    'yield_curve': None,  # Will be set based on data availability
    'credit_spread': None,
    'employment_intensity': None
}
