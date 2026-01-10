"""
Fix all import statements across the project

This script updates all Python files to use the new feature_engineering folder structure.
Run this script from the project root directory.
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def fix_file_imports(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original_content = content
    
    # Fix old-style feature_engineering imports
    replacements = [
        # From feature_engineering import X -> from feature_engineering.feature_engineering import X
        (r'from feature_engineering import (StationaryFeatureEngineer|engineer_features_for_model)',
         r'from feature_engineering.feature_engineering import \1'),
        
        # from feature_utils import X -> from feature_engineering.feature_utils import X
        (r'from feature_utils import',
         r'from feature_engineering.feature_utils import'),
        
        # from feature_config import X -> from feature_engineering.feature_config import X  
        (r'from feature_config import',
         r'from feature_engineering.feature_config import'),
        
        # import feature_engineering -> import feature_engineering.feature_engineering
        (r'^import feature_engineering$',
         r'import feature_engineering.feature_engineering'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Only write if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def find_python_files(directory):
    """Find all Python files in directory."""
    py_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.venv', 'venv']]
        
        for file in files:
            if file.endswith('.py'):
                py_files.append(Path(root) / file)
    
    return py_files


def main():
    """Fix imports in all Python files."""
    print("="*60)
    print("FIXING ALL IMPORT STATEMENTS")
    print("="*60)
    
    # Directories to process
    directories = [
        PROJECT_ROOT / 'first_sample',
        PROJECT_ROOT / 'second_sample',
        PROJECT_ROOT,  # Root level files
    ]
    
    fixed_files = []
    
    for directory in directories:
        if not directory.exists():
            continue
            
        if directory == PROJECT_ROOT:
            # Only process root-level files, not subdirectories
            py_files = [f for f in directory.glob('*.py')]
        else:
            py_files = find_python_files(directory)
        
        for file_path in py_files:
            # Skip this script itself
            if file_path.name == 'fix_all_imports.py':
                continue
                
            if fix_file_imports(file_path):
                fixed_files.append(file_path)
                print(f"  Fixed: {file_path.relative_to(PROJECT_ROOT)}")
    
    print(f"\n Fixed {len(fixed_files)} files")
    
    return fixed_files


if __name__ == "__main__":
    main()
