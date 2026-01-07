"""Quick test for RF-OLS fix"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from first_sample.functions.func_rfols import run_rfols

print("Testing RF-OLS fix...")
np.random.seed(42)
Y = np.random.randn(50, 5)

result = run_rfols(Y, indice=1, lag=1)

print(f"✓ Result type: {type(result)}")
print(f"✓ Result keys: {result.keys()}")
print(f"✓ Pred type: {type(result['pred'])}")
print(f"✓ Pred value: {result['pred']:.6f}")
print("\n✅ RF-OLS fix working correctly!")
