#!/usr/bin/env python3
"""Inherited regression suite plus main runner/evaluator fixtures; no benchmark tests read."""
import sys,unittest,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'app'),str(ROOT/'validation_support'),str(ROOT/'tests')]
suite=unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
suite.addTests(unittest.TestLoader().discover(str(ROOT/'tests_main')))
r=unittest.TextTestRunner(verbosity=2).run(suite)
report={'status':'PASS' if r.wasSuccessful() and not r.skipped else 'INCOMPLETE' if r.wasSuccessful() else 'FAIL',
        'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),
        'skipped_tests':[str(x[0]) for x in r.skipped],'final_test_data_read':False}
if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report));sys.exit(0 if r.wasSuccessful() else 1)
