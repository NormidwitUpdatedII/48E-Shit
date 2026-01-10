"""
Batch update all run files to add RMSE by horizon output (h1-h12)

This script modifies all run files in first_sample/run/ and second_sample/run/
to print RMSE for all 12 horizons in a formatted table.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Files to skip (already updated or special cases)
SKIP_FILES = {'__init__.py', 'run_template.py', 'rf.py'}  # rf.py already updated

# Pattern to match the main loop structure
OLD_PATTERN_1 = re.compile(
    r'(print\(["\']Running.*?["\'].*?\))\s*\n\s*for lag in range\(1, 13\):\s*\n\s*(print.*?lag.*?\n)\s*'
    r'(results\[.*?\] = .*?rolling_window.*?lag=lag\))\s*\n\s*'
    r'(results\[.*?\] = .*?rolling_window.*?lag=lag\))',
    re.DOTALL
)

def update_run_file(file_path, model_name=None):
    """Update a single run file to add RMSE by horizon output."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Skip if already has RMSE by horizon
    if 'RMSE BY HORIZON' in content or 'rmse_cpi' in content:
        return False
    
    # Extract model name from filename
    if model_name is None:
        model_name = file_path.stem.replace('_', ' ').upper()
    
    original_content = content
    
    # Pattern 1: Standard loop pattern
    # Looking for: results = {} followed by loop with rolling_window
    
    # Find the results = {} initialization
    if 'results = {}' not in content:
        print(f"  Skipping {file_path.name}: No 'results = {{}}' found")
        return False
    
    # Add rmse_cpi and rmse_pce dictionaries after results = {}
    content = content.replace(
        'results = {}',
        'results = {}\n    rmse_cpi = {}\n    rmse_pce = {}'
    )
    
    # Update the loop to extract RMSE
    # Pattern: results[f'...{lag}c'] = ..._rolling_window(Y, nprev, indice=1, lag=lag)
    #          results[f'...{lag}p'] = ..._rolling_window(Y, nprev, indice=2, lag=lag)
    
    # Find and update the for loop
    loop_pattern = re.compile(
        r"(for lag in range\(1, 13\):.*?)\n(\s+)(print.*?lag.*?\n)?"
        r"(\s+)(results\[f['\"](\w+)\{lag\}c['\"]\].*?indice=1,.*?lag=lag\))\n"
        r"(\s+)(results\[f['\"](\w+)\{lag\}p['\"]\].*?indice=2,.*?lag=lag\))",
        re.DOTALL
    )
    
    match = loop_pattern.search(content)
    if match:
        model_prefix = match.group(6)  # Extract model prefix like 'ar', 'lasso', etc.
        indent = match.group(2)
        
        new_loop = f"""for lag in range(1, 13):
{indent}print(f"  Running h={{lag}}...")
{indent}results[f'{model_prefix}{{lag}}c'] = {model_prefix}_rolling_window(Y, nprev, indice=1, lag=lag)
{indent}results[f'{model_prefix}{{lag}}p'] = {model_prefix}_rolling_window(Y, nprev, indice=2, lag=lag)
{indent}rmse_cpi[lag] = results[f'{model_prefix}{{lag}}c']['errors']['rmse']
{indent}rmse_pce[lag] = results[f'{model_prefix}{{lag}}p']['errors']['rmse']"""
        
        content = loop_pattern.sub(new_loop, content)
    
    # If content changed, add RMSE summary printing before save
    if content != original_content:
        # Find where forecasts are saved and add RMSE printing before
        save_pattern = re.compile(r'(\n\s+# (?:Combine|Save).*?cpi.*?)', re.IGNORECASE)
        
        rmse_print_block = '''
    # Print RMSE summary by horizon
    print("\\n" + "=" * 60)
    print("RMSE BY HORIZON")
    print("=" * 60)
    print(f"{'Horizon':<10} {'CPI RMSE':<15} {'PCE RMSE':<15}")
    print("-" * 40)
    for h in range(1, 13):
        print(f"h={h:<7} {rmse_cpi[h]:<15.6f} {rmse_pce[h]:<15.6f}")
    
    # Print average RMSE
    avg_cpi = np.mean(list(rmse_cpi.values()))
    avg_pce = np.mean(list(rmse_pce.values()))
    print("-" * 40)
    print(f"{'Average':<10} {avg_cpi:<15.6f} {avg_pce:<15.6f}")
    print("=" * 60)
'''
        
        match = save_pattern.search(content)
        if match:
            content = content[:match.start()] + rmse_print_block + content[match.start():]
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def update_all_run_files():
    """Update all run files in both sample directories."""
    print("=" * 60)
    print("BATCH UPDATE: Adding RMSE by Horizon to All Run Files")
    print("=" * 60)
    
    directories = [
        PROJECT_ROOT / 'first_sample' / 'run',
        PROJECT_ROOT / 'second_sample' / 'run'
    ]
    
    updated_files = []
    skipped_files = []
    
    for directory in directories:
        if not directory.exists():
            continue
        
        print(f"\nProcessing: {directory}")
        print("-" * 40)
        
        for file_path in sorted(directory.glob('*.py')):
            if file_path.name in SKIP_FILES:
                print(f"  Skipping: {file_path.name} (in skip list)")
                skipped_files.append(file_path)
                continue
            
            try:
                if update_run_file(file_path):
                    print(f"  Updated: {file_path.name}")
                    updated_files.append(file_path)
                else:
                    print(f"  No changes: {file_path.name}")
                    skipped_files.append(file_path)
            except Exception as e:
                print(f"  Error: {file_path.name} - {e}")
                skipped_files.append(file_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Updated files: {len(updated_files)}")
    print(f"Skipped files: {len(skipped_files)}")
    
    return updated_files, skipped_files


if __name__ == '__main__':
    updated, skipped = update_all_run_files()
