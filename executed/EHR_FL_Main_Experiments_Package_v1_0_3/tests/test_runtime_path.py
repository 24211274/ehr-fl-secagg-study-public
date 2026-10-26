"""Runtime PATH regression checks. Stub CLIs here are NOT Flower execution."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ehr_fedavg.host import FLOWER_COMMANDS, inspect_flower_command_path, local_environment


class RuntimePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ehr runtime path ')
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'venv with spaces' / 'bin'
        self.bin.mkdir(parents=True)
        self.python = self.bin / 'python'
        self.python.symlink_to(sys.executable)
        self.system = self.root / 'system-bin'
        self.system.mkdir()
        for name in FLOWER_COMMANDS:
            self.make_command(self.bin / name, 'PINNED_RUNTIME_STUB')

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def make_command(path, message):
        path.write_text('#!/bin/sh\nprintf "' + message + '\\n"\n')
        path.chmod(0o755)

    def environment(self, inherited=None):
        if inherited is None:
            inherited = str(self.system)
        with patch.dict(os.environ, {'PATH': inherited}), patch('sys.executable', str(self.python)):
            return local_environment(self.root / 'flower_state')

    def inspect(self, env):
        with patch('sys.executable', str(self.python)):
            return inspect_flower_command_path(env)

    def test_selected_environment_bin_is_first(self):
        env = self.environment()
        self.assertEqual(env['PATH'].split(os.pathsep)[0], str(self.bin))
        self.assertEqual(env['PATH'].split(os.pathsep)[1:], [str(self.system)])

    def test_parent_environment_not_modified(self):
        before = dict(os.environ)
        self.environment()
        self.assertEqual(dict(os.environ), before)

    def test_four_commands_include_superexec(self):
        self.assertEqual(set(FLOWER_COMMANDS), {'flwr','flower-superlink','flower-supernode','flower-superexec'})
        report = self.inspect(self.environment())
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(len(report['commands']), 4)
        self.assertFalse(report['services_started_by_this_check'])

    def test_missing_superexec_rejected(self):
        (self.bin/'flower-superexec').unlink()
        report = self.inspect(self.environment())
        self.assertEqual(report['status'],'FAIL')
        self.assertIn('Missing executable', report['errors'][0])

    def test_non_executable_superexec_rejected(self):
        (self.bin/'flower-superexec').chmod(0o644)
        self.assertEqual(self.inspect(self.environment())['status'], 'FAIL')

    def test_wrong_global_fallback_not_accepted(self):
        (self.bin/'flower-superexec').unlink()
        self.make_command(self.system/'flower-superexec', 'WRONG_ENVIRONMENT')
        self.assertEqual(self.inspect(self.environment())['status'], 'FAIL')

    def test_existing_global_command_loses_to_pinned(self):
        self.make_command(self.system/'flower-superexec', 'WRONG_ENVIRONMENT')
        env = self.environment()
        self.assertEqual(shutil.which('flower-superexec', path=env['PATH']), str(self.bin/'flower-superexec'))
        self.assertEqual(self.inspect(env)['status'],'PASS')

    def test_python_symlink_does_not_change_bin_directory(self):
        self.assertNotEqual(self.python.resolve().parent, self.bin)
        self.assertEqual(self.inspect(self.environment())['status'],'PASS')

    def test_empty_inherited_path_supported(self):
        env=self.environment('')
        self.assertEqual(env['PATH'],str(self.bin))
        self.assertEqual(self.inspect(env)['status'],'PASS')

    def test_prepending_is_idempotent(self):
        env=self.environment(str(self.bin)+os.pathsep+str(self.system)+os.pathsep+str(self.bin))
        self.assertEqual(env['PATH'].split(os.pathsep).count(str(self.bin)),1)

    def test_spaces_in_command_directory_are_preserved(self):
        env=self.environment()
        result=subprocess.run(['flower-superexec','--help'], env=env, capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0)
        self.assertEqual(result.stdout.strip(),'PINNED_RUNTIME_STUB')

    def test_bare_subprocess_lookup_reproduced_and_fixed(self):
        # The old launcher gave the parent an absolute executable but passed a
        # PATH that could not locate a child command by its bare name.
        parent_code="import subprocess; subprocess.run(['flower-superexec','--help'],check=True)"
        bad_env=dict(os.environ,PATH=str(self.system))
        old=subprocess.run([sys.executable,'-c',parent_code],env=bad_env,capture_output=True,text=True,timeout=10)
        self.assertNotEqual(old.returncode,0)
        self.assertIn('FileNotFoundError',old.stderr)
        new=subprocess.run([sys.executable,'-c',parent_code],env=self.environment(),capture_output=True,text=True,timeout=10)
        self.assertEqual(new.returncode,0,new.stderr)
        self.assertEqual(new.stdout.strip(),'PINNED_RUNTIME_STUB')

    def test_path_inspection_does_not_execute_commands(self):
        marker=self.root/'must_not_be_written'
        target=self.bin/'flower-superexec'
        target.write_text('#!/bin/sh\ntouch "'+str(marker)+'"\n');target.chmod(0o755)
        self.assertEqual(self.inspect(self.environment())['status'],'PASS')
        self.assertFalse(marker.exists())

    def test_arbitrary_current_directory_does_not_supply_missing_command(self):
        (self.bin/'flower-superexec').unlink()
        self.make_command(self.root/'flower-superexec','WRONG_WORKDIR')
        self.assertEqual(self.inspect(self.environment())['status'],'FAIL')

if __name__=='__main__':unittest.main()
