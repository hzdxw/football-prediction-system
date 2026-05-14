#!/usr/bin/env python3
"""Import chain check for core prediction modules."""
import importlib.util, sys, os

def try_import(name, path):
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        print('  OK  ' + name)
        return True
    except Exception as e:
        print('  FAIL ' + name + ': ' + str(e))
        return False

files = {
    '_cp': 'prediction/calibrated_poisson.py',
    '_pa': 'prediction/post_adjust.py',
    '_vb': 'prediction/value_bet.py',
}
for name, path in files.items():
    if os.path.exists(path):
        sys.modules[name] = None
        try_import(name, path)
