"""CPU checks for subset routing, completeness, and deterministic episode prefixes."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh'


class SubsetEvaluationTests(unittest.TestCase):
    def test_task_selection_and_validation(self):
        source = LAUNCHER.read_text().split('# ===== Common environment =====')[0]
        source += '\nprintf "SELECTED=%s EPISODES=%s SLOTS=%s\\n" "${task_queue[*]}" "$test_num" "$num_slots"\n'
        valid = subprocess.run(['bash', '-c', source, 'launcher', '--tasks', 'shake_bottle,click_bell',
                                '--test_num', '30', '--num_gpus', '4', '--num_per_gpu', '2'], capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertIn('SELECTED=shake_bottle click_bell EPISODES=30 SLOTS=2', valid.stdout)
        for flags in (['--tasks', 'unknown'], ['--tasks', 'click_bell,click_bell'],
                      ['--tasks', 'click_bell,'], ['--test_num', '0'], ['--test_num', '101'],
                      ['--test_num', 'abc'], ['--num_tasks', '0']):
            bad = subprocess.run(['bash', '-c', source, 'launcher', *flags], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0, flags)
        default = subprocess.run(['bash', '-c', source, 'launcher'], capture_output=True, text=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertIn('Tasks this run (50)', default.stdout)
        self.assertIn('EPISODES=100', default.stdout)

    def test_thirty_episodes_match_full_prefix(self):
        # Execute the actual simulation loop with a deterministic lightweight environment.
        # Include invalid expert seeds to check that K counts valid episodes, not raw seeds.
        for relative in ('RoboTwin/script/eval_policy_client_lingbotvla.py',
                         'lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py'):
            tree = ast.parse((ROOT / relative).read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'eval_policy')
            calls = []
            def descriptions(task, episodes, budget):
                calls.append(budget)
                return [{'unseen': [f'instruction-{random.randrange(100000)}' for _ in range(budget)]}]
            namespace = {'np': types.SimpleNamespace(random=types.SimpleNamespace(choice=random.choice)),
                         'os': os, 'generate_episode_descriptions': descriptions, 'UnStableError': RuntimeError}
            exec(compile(ast.Module(body=[function], type_ignores=[]), relative, 'exec'), namespace)
            class Environment:
                eval_video_path = None
                render_freq = 0
                step_lim = 0
                take_action_cnt = 0
                def __init__(self):
                    self.episodes = []
                def setup_demo(self, seed, **kwargs):
                    self.seed = seed
                    self.plan_success = seed % 7 != 0
                    random.seed(seed)
                def play_once(self):
                    return {'info': {}}
                def close_env(self, **kwargs):
                    pass
                def check_success(self):
                    return self.plan_success
                def set_instruction(self, instruction):
                    self.episodes.append((self.seed, instruction))
            def run(count):
                env = Environment()
                args = dict(task_name='fake', policy_name='ACT', task_config='demo_clean',
                            ckpt_setting='fake', render_freq=0, clear_cache_freq=5)
                with contextlib.redirect_stdout(io.StringIO()):
                    namespace['eval_policy']('fake', env, args, types.SimpleNamespace(infer=lambda _: {}),
                                             100000, test_num=count, instruction_type='unseen', usr_args={'robo_name': 'robotwin'})
                return env.episodes
            full, short = run(100), run(30)
            self.assertEqual(len(short), 30)
            self.assertEqual(short, full[:30], relative)
            self.assertTrue(all(budget == 100 for budget in calls))

    def test_baseline_routing_and_result_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ('baseline/eval.sh', 'baseline/make_results.py', '灵波2026/初赛评测结果模板_JSON版.json'):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
            checkpoint = root / 'baseline/work/train_output/checkpoints/global_step_2000/hf_ckpt'
            checkpoint.mkdir(parents=True)
            (checkpoint.parents[2] / 'lingbotvla_cli.yaml').touch()
            qwen = root / 'qwen'; qwen.mkdir(); (qwen / 'config.json').write_text('{}')
            conda = root / 'conda.sh'; conda.touch()
            binary = root / 'bin'; binary.mkdir()
            nvidia = binary / 'nvidia-smi'; nvidia.write_text('#!/bin/bash\necho GPU\n'); nvidia.chmod(0o755)
            launcher = root / 'lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh'
            launcher.parent.mkdir(parents=True)
            launcher.write_text('''#!/bin/bash
python - "$@" <<'STUB'
import sys
from pathlib import Path
flags = dict(zip(sys.argv[1::2], sys.argv[2::2]))
run = Path(flags['--output_base']) / 'fake_run'
(run / 'eval_logs').mkdir(parents=True)
k = int(flags['--test_num'])
(run / 'stats.txt').write_text(f'All tasks fully executed {k} episodes\\n')
tasks = flags.get('--tasks')
if tasks is None:
    import json
    tasks = ','.join(json.loads((Path.cwd().parent / '灵波2026/初赛评测结果模板_JSON版.json').read_text())['results']['clean'])
for task in tasks.split(','):
    (run / 'eval_logs' / f'{task}.log').write_text(f'Success rate: 1/{k} => {100/k}%\\n')
STUB
''')
            submission = root / 'baseline/work/submission/01_评测结果/results.json'
            submission.parent.mkdir(parents=True); submission.write_text('previous-full-result')
            env = {**os.environ, 'PATH': f'{binary}:{os.environ["PATH"]}', 'TEAM_ID': '1382900',
                   'QWEN3VL_PATH': str(qwen), 'CONDA_SH': str(conda), 'EVAL_COMPAT': '0',
                   'EVAL_TASKS': 'recommended8', 'EVAL_EPISODES': '30'}
            command = ['bash', str(root / 'baseline/eval.sh'), str(checkpoint)]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            subset = next((root / 'baseline/work/eval').glob('*/results.json'))
            payload = json.loads(subset.read_text())
            self.assertEqual(len(payload['results']['clean']), 8)
            self.assertTrue(all(v['attempts'] == 30 for v in payload['results']['randomized'].values()))
            self.assertEqual(payload['evaluation']['episode_selection'], 'first_valid_episodes')
            self.assertEqual(submission.read_text(), 'previous-full-result')
            env.update(EVAL_TASKS='all', EVAL_EPISODES='100')
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            full = json.loads(submission.read_text())
            self.assertEqual(len(full['results']['clean']), 50)
            self.assertNotIn('evaluation', full)
            # A completion marker alone must not allow incomplete task logs.
            spec = importlib.util.spec_from_file_location('results', ROOT / 'baseline/make_results.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            clean_run = subset.parent / 'clean/fake_run'
            (clean_run / 'stats.txt').write_text('All tasks fully executed 30 episodes\n')
            (clean_run / 'eval_logs/click_bell.log').write_text('Success rate: 1/29 => 3.4%\n')
            with self.assertRaises(ValueError):
                module.collect(clean_run, ['click_bell'], 30)


if __name__ == '__main__':
    unittest.main()
