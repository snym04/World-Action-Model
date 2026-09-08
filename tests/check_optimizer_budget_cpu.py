"""CPU unit check using the actual training loop extracted without Wan imports."""
import ast
from contextlib import nullcontext
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path
import torch
from torch.utils.data import DataLoader

torch.set_num_threads(1)
source = Path(sys.argv[1])
tree = ast.parse(source.read_text())
names = {'_build_lr_scheduler', '_save_full_state', '_consume_swanlab_manifest_env', 'launch_training_task'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(nodes) == 4
logs = []
sys.modules['swanlab'] = types.SimpleNamespace(
    init=lambda **kw: types.SimpleNamespace(id='cpu-unit-test'),
    log=lambda values, step: logs.append((step, values)), finish=lambda: None)
ns = dict(torch=torch, math=math, os=os, json=json, DataLoader=DataLoader,
          tqdm=lambda it, **kw: it, dual_stream_action_collate_fn=lambda xs: torch.stack(xs))
ns.update(vars(torch.optim.lr_scheduler))
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
    def trainable_modules(self):
        return self.parameters()
    def forward(self, batch):
        loss = ((self.weight * batch - 1) ** 2).mean()
        return {k: loss for k in ('loss', 'loss_rgb', 'loss_flow', 'loss_action', 'loss_video')}

class Logger:
    def __init__(self, output):
        self.output_path = str(output)
        self.num_steps = 0
        self.saved = []
    def on_step_end(self, accelerator, model, save_steps):
        self.num_steps += 1
        if save_steps and self.num_steps % save_steps == 0:
            self.saved.append(self.num_steps)
    def on_training_end(self, accelerator, model, save_steps):
        if self.num_steps not in self.saved:
            self.saved.append(self.num_steps)
    def on_epoch_end(self, *args):
        pass

arg_names = {n.attr for node in nodes for n in ast.walk(node)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'args'}

def run(output, maximum, resume=None):
    kwargs = dict.fromkeys(arg_names)
    kwargs.update(learning_rate=0.01, weight_decay=0.0, dataset_num_workers=0,
                  save_steps=2, num_epochs=1, max_train_steps=maximum,
                  gradient_accumulation_steps=2, find_unused_parameters=False,
                  save_every_n_epochs=100, load_from_cache=False, batch_size=2,
                  seed=42, lr_max_steps=4, lr_warmup_steps=1, lr_warmup_ratio=0,
                  lr_scheduler_type='cosine', full_state_keep=4,
                  resume_state_dir=str(resume) if resume else None,
                  dataset_type='unit', output_path=str(output))
    model = Model()
    logger = Logger(output)
    ns['launch_training_task']([torch.tensor(float(i+1)) for i in range(10)],
                               model, logger, args=types.SimpleNamespace(**kwargs))
    state = json.loads((output / f'state/step-{maximum}/trainer_state.json').read_text())
    assert state['global_step'] == maximum
    assert logger.num_steps == maximum and maximum in logger.saved
    return model.weight.detach().clone(), state

test_dir = os.environ.get('WAM_CPU_TEST_DIR')
context = nullcontext(test_dir) if test_dir else tempfile.TemporaryDirectory(
    dir='/home/zmh/WAM/tmp', prefix='budget-unit-')
with context as tmp:
    base = Path(tmp)
    full, full_state = run(base/'full', 4)
    _, partial_state = run(base/'partial', 2)
    resumed, resumed_state = run(base/'resumed', 4, base/'partial/state/step-2')
    assert partial_state['micro_step'] == 4
    assert full_state['micro_step'] == resumed_state['micro_step'] == 8
    assert torch.allclose(full, resumed, atol=1e-7, rtol=0), (full, resumed)
    # Non-cadence final step must still produce full state.
    _, odd_state = run(base/'odd', 3)
    assert odd_state['micro_step'] == 6
print('PASS: optimizer budget, accumulation boundary, padded-epoch resume, final full state, identical resumed weights')
