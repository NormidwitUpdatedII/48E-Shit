from utils import load_csv
import numpy as np

Y = load_csv('first_sample/rawdata.csv')
print(f'Shape: {Y.shape}')
print(f'Total NaN: {np.isnan(Y).sum()}')

nan_per_col = np.isnan(Y).sum(axis=0)
print(f'Max NaN in a column: {nan_per_col.max()}')
print(f'Columns with all NaN: {(nan_per_col == Y.shape[0]).sum()}')
print(f'Columns with some NaN: {(nan_per_col > 0).sum()}')

# Check first window
nprev = 132
Y_window = Y[(nprev - nprev):(Y.shape[0] - nprev), :]
print(f'\nFirst window shape: {Y_window.shape}')
print(f'First window NaN: {np.isnan(Y_window).sum()}')
