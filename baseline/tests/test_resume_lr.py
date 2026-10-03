"""Exercise the real Torch scheduler without importing training/model dependencies."""
import ast
import copy
import math
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'lingbot-vla-v2/lingbotvla/optim/lr_scheduler.py'
module = ast.parse(SOURCE.read_text())
module.body = [node for node in module.body
               if not (isinstance(node, ast.ImportFrom) and node.level)
               and not (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'logger' for t in node.targets))]
namespace = {}
exec(compile(module, str(SOURCE), 'exec'), namespace)
build = namespace['build_lr_scheduler']
resume = namespace['resume_lr_scheduler']


class ResumeLearningRateTests(unittest.TestCase):
    def optimizer(self, base=1e-4):
        return torch.optim.SGD([{'params': [torch.nn.Parameter(torch.ones(1))], 'lr': base},
                                {'params': [torch.nn.Parameter(torch.ones(1))], 'lr': base * 3}], momentum=0.9)

    def checkpoint(self):
        optimizer = self.optimizer()
        scheduler = build(optimizer, train_steps=2000, lr=1e-4, lr_min=5e-5, lr_decay_style='cosine')
        for _ in range(2000):
            for group in optimizer.param_groups:
                group['params'][0].grad = torch.ones(1)
            optimizer.step()
            scheduler.step()
        return copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict())

    def test_extended_horizon_applies_before_first_update(self):
        checkpoint, _ = self.checkpoint()
        optimizer = self.optimizer()
        scheduler = build(optimizer, train_steps=50000, lr=1e-4, lr_min=5e-5, lr_decay_style='cosine')
        optimizer.load_state_dict(checkpoint)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-5)
        momentum = optimizer.param_groups[0]['params'][0]
        before = optimizer.state[momentum]['momentum_buffer'].clone()
        resume(scheduler, 2000)
        expected = 5e-5 + 5e-5 * (1 + math.cos(math.pi * 2000 / 50000)) / 2
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], expected)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], expected * 3)
        self.assertTrue(torch.equal(optimizer.state[momentum]['momentum_buffer'], before))
        self.assertEqual(scheduler.last_epoch, 2000)
        optimizer.step(); scheduler.step()
        self.assertEqual(scheduler.last_epoch, 2001)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 5e-5 + 5e-5 * (1 + math.cos(math.pi * 2001 / 50000)) / 2)
        resume(scheduler, 50000)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 5e-5)
        self.assertAlmostEqual(scheduler.get_last_lr()[1], 15e-5)

    def test_unchanged_config_matches_normal_restore(self):
        checkpoint, schedule_state = self.checkpoint()
        optimizers = [self.optimizer(), self.optimizer()]
        schedulers = [build(o, train_steps=2000, lr=1e-4, lr_min=5e-5, lr_decay_style='cosine') for o in optimizers]
        for optimizer in optimizers:
            optimizer.load_state_dict(checkpoint)
        schedulers[0].load_state_dict(schedule_state)
        resume(schedulers[1], 2000)
        self.assertEqual(schedulers[0].state_dict(), schedulers[1].state_dict())
        for optimizer, scheduler in zip(optimizers, schedulers):
            optimizer.step(); scheduler.step()
        self.assertEqual(schedulers[0].get_last_lr(), schedulers[1].get_last_lr())

    def test_new_base_lr_and_warmup_are_used(self):
        checkpoint, _ = self.checkpoint()
        optimizer = self.optimizer(base=2e-4)
        scheduler = build(optimizer, train_steps=50000, lr=2e-4, lr_min=5e-5,
                          lr_decay_style='cosine', lr_warmup_ratio=0.1)
        optimizer.load_state_dict(checkpoint)
        resume(scheduler, 2000)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 2e-4 * 2000 / 5000)
        self.assertAlmostEqual(optimizer.param_groups[0]['initial_lr'], 2e-4)
        self.assertAlmostEqual(optimizer.param_groups[1]['initial_lr'], 6e-4)


if __name__ == '__main__':
    unittest.main()
