#!/usr/bin/env python3
"""AST syntax check for core prediction modules."""
import ast, sys, os

files = [
    'ml_predict_5play.py',
    'prediction/post_adjust.py',
    'prediction/calibrated_poisson.py',
    'prediction/value_bet.py',
    'prediction/strategies/super_fusion.py',
    'prediction/ensemble_predict.py',
]

errors = []
for f in files:
    if not os.path.exists(f):
        errors.append('MISSING: ' + f)
        continue
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print('  OK  ' + f)
    except SyntaxError as e:
        errors.append('SYNTAX ' + f + ':' + str(e.lineno) + ' ' + e.msg)

if errors:
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1)
else:
    print('All files passed AST check')
