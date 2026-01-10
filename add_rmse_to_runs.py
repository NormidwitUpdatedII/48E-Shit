"""
Minimal RMSE Insertion Script
Adds RMSE extraction and summary printing to run files with minimal changes.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Skip these files
SKIP_FILES = {'__init__.py', 'run_template.py', 'rf.py'}  # rf.py already done


def add_rmse_to_file(file_path):
    """Add minimal RMSE extraction to a run file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Skip if already has RMSE extraction
    if 'RMSE BY HORIZON' in content or "['errors']['rmse']" in content:
        return False, "Already has RMSE"
    
    original = content
    
    # Step 1: Find the model prefix from the results assignment
    # Pattern: results[f'{prefix}{lag}c'] = ...
    prefix_match = re.search(r"results\[f['\"](\w+)\{lag\}c['\"]\]", content)
    if not prefix_match:
        return False, "No results pattern found"
    
    prefix = prefix_match.group(1)
    
    # Step 2: Add RMSE dictionaries after "results = {}"
    if 'results = {}' in content:
        content = content.replace(
            'results = {}',
            f'results = {{}}\n    rmse_cpi = {{}}\n    rmse_pce = {{}}'
        )
    
    # Step 3: Add RMSE extraction after each model call
    # Find: results[f'{prefix}{lag}p'] = ..._rolling_window(..., lag=lag)
    # Add: rmse_cpi[lag] = results[f'{prefix}{lag}c']['errors']['rmse']
    #      rmse_pce[lag] = results[f'{prefix}{lag}p']['errors']['rmse']
    
    pattern = rf"(results\[f'{prefix}\{{lag\}}p'\] = \w+_rolling_window\([^)]+\))"
    
    def add_rmse_lines(match):
        original_line = match.group(1)
        return f"""{original_line}
        rmse_cpi[lag] = results[f'{prefix}{{lag}}c']['errors']['rmse']
        rmse_pce[lag] = results[f'{prefix}{{lag}}p']['errors']['rmse']"""
    
    content = re.sub(pattern, add_rmse_lines, content)
    
    # Step 4: Add RMSE summary print before "return results"
    rmse_summary = f'''
    # Print RMSE by horizon
    print("\\nRMSE BY HORIZON:")
    print(f"{{'Horizon':<8}} {{'CPI':<12}} {{'PCE':<12}}")
    for h in range(1, 13):
        print(f"h={{h:<6}} {{rmse_cpi.get(h, 0):<12.6f}} {{rmse_pce.get(h, 0):<12.6f}}")
    print(f"Average: {{np.mean(list(rmse_cpi.values())):.6f}}  {{np.mean(list(rmse_pce.values())):.6f}}")
'''
    
    content = re.sub(r'(\n\s+return results)', rmse_summary + r'\1', content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "Updated"
    
    return False, "No changes made"


def process_all_files():
    """Process all run files in both sample directories."""
    dirs = [
        PROJECT_ROOT / 'first_sample' / 'run',
        PROJECT_ROOT / 'second_sample' / 'run'
    ]
    
    updated = []
    skipped = []
    
    for d in dirs:
        if not d.exists():
            continue
        
        print(f"\n{d}:")
        for f in sorted(d.glob('*.py')):
            if f.name in SKIP_FILES:
                print(f"  SKIP: {f.name}")
                skipped.append(f.name)
                continue
            
            success, msg = add_rmse_to_file(f)
            status = "OK" if success else "SKIP"
            print(f"  {status}: {f.name} ({msg})")
            
            if success:
                updated.append(f.name)
            else:
                skipped.append(f.name)
    
    print(f"\n\nSummary: {len(updated)} updated, {len(skipped)} skipped")
    return updated


if __name__ == '__main__':
    process_all_files()
