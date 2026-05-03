# OFA Modernization Migration Guide
## Python 3.12 + PyTorch 2.x (tested: PyTorch 2.9.1, torchvision 0.24.1)

This document records every significant change made to the `once-for-all` repository
to bring it from the original PyTorch 1.4/Horovod era to a modern Python 3.12 +
PyTorch 2.x stack. Use it as a recipe when migrating any codebase that shares OFA
as a base (e.g. ComPoFA, AttentiveNAS, etc.).

---

## 1. Dependency overhaul

### Old requirements
```
Python 3.6+
PyTorch 1.4.0+
Horovod
```

### New requirements
```
Python 3.12+ (tested 3.12.13)
PyTorch 2.9.1+
torchvision 0.24.1+
tqdm, filelock, gdown, pillow
```

**Horovod is completely removed.** All distributed training now uses
`torch.distributed` (NCCL backend) launched via `torchrun`.

### Conda environment setup
```bash
conda create -n ofa python=3.12 -y
conda activate ofa
pip install torch==2.9.1 torchvision==0.24.1
pip install filelock gdown tqdm pillow
pip install -e .
```

---

## 2. `setup.py` — version string format

**Change**: The version suffix was changed from `_YYYYMMDDHHM` to `.devYYYYMMDDHHM`
to comply with PEP 440 (pip rejects non-compliant version strings in newer Python).

```python
# Before
VERSION += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")

# After
VERSION += ".dev" + datetime.datetime.now().strftime("%Y%m%d%H%M")
```

---

## 3. `ofa/utils/my_dataloader/my_data_loader.py`

Three private PyTorch internals that were imported directly no longer exist in
PyTorch 2.x.

### 3a. `torch._six` removed (PyTorch 2.0+)
`torch._six` was a compatibility shim for Python 2/3. It was removed entirely in
PyTorch 2.0.

```python
# Before
from torch._six import string_classes

# After — define it directly (Python 3 only)
string_classes = (str,)
```

### 3b. `torch.multiprocessing.Queue` alias removed
```python
# Before
from torch.multiprocessing import Queue as queue

# After — use stdlib directly
import queue
```

### 3c. `ExceptionWrapper` location changed
```python
# Before
from torch._utils import ExceptionWrapper

# After — with fallback for different PyTorch versions
try:
    from torch._utils import ExceptionWrapper
except ImportError:
    from torch.utils.data._utils.worker import ExceptionWrapper
```

### 3d. Pin-memory thread needs explicit device type argument
```python
# Before
torch.utils.data._utils.pin_memory._pin_memory_loop(
    self._worker_result_queue,
    self._data_queue,
    torch.cuda.current_device(),
    self._pin_memory_thread_done_event,
)

# After — PyTorch 2.x _pin_memory_loop requires a device_type string
torch.utils.data._utils.pin_memory._pin_memory_loop(
    self._worker_result_queue,
    self._data_queue,
    torch.cuda.current_device(),
    self._pin_memory_thread_done_event,
    "cuda",                          # <-- new required arg
)
```

---

## 4. `ofa/utils/my_dataloader/my_data_worker.py`

Same family of removals as the loader file above.

```python
# Before
from torch.multiprocessing import Queue as queue
from torch._utils import ExceptionWrapper
from torch.utils.data._utils import (
    signal_handling,
    MP_STATUS_CHECK_INTERVAL,
    IS_WINDOWS,
)

# After
import queue          # stdlib
import sys
from torch.utils.data._utils import signal_handling

try:
    from torch._utils import ExceptionWrapper
except ImportError:
    from torch.utils.data._utils.worker import ExceptionWrapper

try:
    from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
except ImportError:
    MP_STATUS_CHECK_INTERVAL = 5.0   # fallback default

IS_WINDOWS = sys.platform == "win32" # was removed from torch internals
```

---

## 5. `ofa/utils/my_dataloader/my_random_resize_crop.py`

PIL integer interpolation constants (`Image.BILINEAR`, etc.) were deprecated and
then removed from the `torchvision` transform API. Replace them everywhere with
`torchvision.transforms.InterpolationMode` enum values.

```python
# Before
from PIL import Image
_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    ...
}

class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, ..., interpolation=Image.BILINEAR):
        ...

class MyResizeRandomCrop(object):
    def __init__(self, interpolation=Image.BILINEAR, ...):
        ...

class MyResize(object):
    def __init__(self, interpolation=Image.BILINEAR):
        ...
```

```python
# After
from torchvision.transforms import InterpolationMode

_interpolation_to_str = {
    InterpolationMode.NEAREST: "InterpolationMode.NEAREST",
    InterpolationMode.BILINEAR: "InterpolationMode.BILINEAR",
    ...
}

class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, ..., interpolation=InterpolationMode.BILINEAR):
        ...
# (same for MyResizeRandomCrop and MyResize)
```

Also update `__repr__` methods that index into the old dict to use `.get()` with a
safe fallback:
```python
# Before
interpolate_str = _pil_interpolation_to_str[self.interpolation]

# After
interpolate_str = _interpolation_to_str.get(self.interpolation, str(self.interpolation))
```

---

## 6. `ofa/utils/common_tools.py` — `DistributedMetric` and `DistributedTensor`

Replace every `horovod.torch` call with `torch.distributed` equivalents, guarded by
`is_initialized()` so single-GPU runs still work.

### `DistributedMetric.update()`
```python
# Before
import horovod.torch as hvd
val *= delta_n
self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)

# After
val = val.detach().clone() * delta_n
if torch.distributed.is_initialized():
    torch.distributed.all_reduce(val)
    val /= torch.distributed.get_world_size()
self.sum += val.cpu()
```

### `DistributedTensor.avg`
```python
# Before
import horovod.torch as hvd
if not self.synced:
    self.sum = hvd.allreduce(self.sum, name=self.name)

# After
if not self.synced:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(self.sum)
        self.sum /= torch.distributed.get_world_size()
```

---

## 7. `ofa/utils/pytorch_utils.py` — `LogSoftmax` dimension

`nn.LogSoftmax()` without an explicit `dim` argument raises an error in PyTorch 2.x
(it was only a warning in 1.x).

```python
# Before
logsoftmax = nn.LogSoftmax()

# After
logsoftmax = nn.LogSoftmax(dim=1)
```

---

## 8. `ofa/imagenet_classification/run_manager/distributed_run_manager.py`

This is the largest change. The entire Horovod distributed training stack is replaced
with `torch.distributed`.

### 8a. New `_SyncedOptimizer` wrapper (replaces `hvd.DistributedOptimizer`)

Add this class at the top of the file. It wraps any standard PyTorch optimizer and
manually all-reduces gradients before each `.step()` call, replicating what
`hvd.DistributedOptimizer` did.

```python
import torch.distributed as dist

class _SyncedOptimizer:
    """Wraps a standard optimizer and all-reduces gradients before each step."""

    def __init__(self, optimizer, net):
        self._optimizer = optimizer
        self._net = net

    def zero_grad(self, set_to_none=False):
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param_group in self._optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size
        return self._optimizer.step(closure)

    def state_dict(self):        return self._optimizer.state_dict()
    def load_state_dict(self, s): self._optimizer.load_state_dict(s)

    @property
    def param_groups(self): return self._optimizer.param_groups
```

### 8b. `DistributedRunManager.__init__` — signature and optimizer construction

```python
# Before
def __init__(self, path, net, run_config, hvd_compression, ...):
    import horovod.torch as hvd
    ...
    self.optimizer = self.run_config.build_optimizer(net_params)
    self.optimizer = hvd.DistributedOptimizer(
        self.optimizer,
        named_parameters=self.net.named_parameters(),
        compression=hvd_compression,
        backward_passes_per_step=backward_steps,
    )

# After
def __init__(self, path, net, run_config,
             hvd_compression=None,  # kept for API compat, ignored
             backward_steps=1, is_root=False, init=True):
    # no horovod import
    ...
    base_optimizer = self.run_config.build_optimizer(net_params)
    self.optimizer = _SyncedOptimizer(base_optimizer, self.net)
```

> Any call site that passes `hvd_compression` positionally or as a keyword still
> works — the parameter is accepted and silently ignored.

### 8c. `load_model` — `torch.load` safety flag + all-rank loading

In PyTorch 2.x, `torch.load` requires `weights_only` to be explicitly set.
Also, the original code only loaded the checkpoint on rank 0 (inside
`if self.is_root:`); with `torch.distributed` all ranks must load (or you
broadcast manually). The simplest fix is to load on all ranks.

```python
# Before
checkpoint = torch.load(model_fname, map_location="cpu")

# After
checkpoint = torch.load(model_fname, map_location="cpu", weights_only=False)
```

Remove the outer `if self.is_root:` guard on the entire load block (keep it only
around the `print` statement).

### 8d. `broadcast` — replace `hvd.broadcast_*` with `dist.broadcast_*`

```python
# Before
import horovod.torch as hvd
self.start_epoch = hvd.broadcast(torch.LongTensor(1).fill_(self.start_epoch)[0], 0, name="start_epoch").item()
self.best_acc    = hvd.broadcast(torch.Tensor(1).fill_(self.best_acc)[0], 0, name="best_acc").item()
hvd.broadcast_parameters(self.net.state_dict(), 0)
hvd.broadcast_optimizer_state(self.optimizer, 0)

# After
if not dist.is_initialized():
    return
state = [self.start_epoch, self.best_acc]
dist.broadcast_object_list(state, src=0)
self.start_epoch, self.best_acc = int(state[0]), float(state[1])

for param in self.net.parameters():
    dist.broadcast(param.data, src=0)
for buf in self.net.buffers():
    dist.broadcast(buf, src=0)
```

---

## 9. `ofa/imagenet_classification/elastic_nn/training/progressive_shrinking.py`

### 9a. `_unwrap_net()` helper — handle DDP as well as DataParallel

The original code only unwrapped `nn.DataParallel`. With `torch.distributed` the
model is wrapped in `DistributedDataParallel` instead. Add a single helper and use
it everywhere:

```python
def _unwrap_net(net):
    """Unwrap DataParallel or DistributedDataParallel to get the base module."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(net, (nn.DataParallel, DDP)):
        return net.module
    return net
```

Replace every occurrence of:
```python
dynamic_net = run_manager.net
if isinstance(dynamic_net, nn.DataParallel):
    dynamic_net = dynamic_net.module
```
with:
```python
dynamic_net = _unwrap_net(run_manager.net)
```

Affected functions: `validate`, `train_one_epoch`, `train_elastic_depth`,
`train_elastic_expand`, `train_elastic_width_mult`.

### 9b. `load_models` — `weights_only=False`

```python
# Before
init = torch.load(model_path, map_location="cpu")["state_dict"]

# After
init = torch.load(model_path, map_location="cpu", weights_only=False)["state_dict"]
```

### 9c. Teacher-model guard in `train_one_epoch`

The original code called `args.teacher_model.train()` whenever `args.kd_ratio > 0`,
even when `args.teacher_model` is `None`. Fix both the soft-target computation and
the loss branch:

```python
# Before
if args.kd_ratio > 0:
    args.teacher_model.train()
    ...

if args.kd_ratio == 0:
    loss = ...ce...
else:
    loss = ...kd...

# After
if args.kd_ratio > 0 and args.teacher_model is not None:
    args.teacher_model.train()
    ...

if args.kd_ratio == 0 or args.teacher_model is None:
    loss = ...ce...
else:
    loss = ...kd...
```

---

## 10. New files added

### `train_ofa_resnet.py`

A complete from-scratch training script for the OFA-ResNet50 supernet. Key design
points that ComPoFA scripts should mirror:

- Uses `torchrun` / `torch.distributed` for multi-GPU, falls back gracefully to
  single-GPU when `RANK`/`WORLD_SIZE` env vars are absent.
- Distributed setup:
  ```python
  if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
      dist.init_process_group(backend="nccl")
      local_rank = int(os.environ["LOCAL_RANK"])
      torch.cuda.set_device(local_rank)
      is_root = dist.get_rank() == 0
      num_gpus = dist.get_world_size()
  else:
      local_rank = 0; is_root = True; num_gpus = 1
      torch.cuda.set_device(0)
  ```
- Always calls `dist.destroy_process_group()` at the end.
- ImageNet path is passed via `--imagenet_path` CLI flag (sets
  `ImagenetDataProvider.DEFAULT_PATH`).
- `DistributedRunManager` is called **without** `hvd_compression`.
- Three progressive-shrinking stages (`expand`, `width`, `depth`), each with two
  phases; all hyper-parameters are hard-coded in the script rather than in a
  separate config file.
- `weights_only=False` on every `torch.load` call.

### `run_ofa_resnet_training.sh`

Bash orchestration script for the full six-phase training pipeline:

- Runs phases in order: expand/1 → expand/2 → width/1 → width/2 → depth/1 → depth/2.
- **Skips** any phase whose `model_best.pth.tar` already exists (resume-friendly).
  Pass `--force` to override.
- Chains checkpoint paths automatically between phases.
- Warns (but does not abort) if the seed pretrained checkpoint is missing.
- Prints a timing summary after each phase and a total at the end.
- CLI flags: `--imagenet_path`, `--nproc_per_node` (default 8), `--checkpoint_dir`,
  `--force`.

---

## 11. README.md changes

The following sections were added or replaced:

| Section | Change |
|---|---|
| **Environment Setup** | New section: conda env creation, pip install commands, note that Horovod is no longer required |
| **How to train OFA Networks** | Renamed to **"How to train OFA-ResNet50 supernet"**; replaced `mpirun`/`horovodrun` commands with `torchrun` equivalents; added docs for `run_ofa_resnet_training.sh` (automated) and `train_ofa_resnet.py` (manual/resume) |
| **How to train OFA-MobileNetV3 supernet** | New sub-section noting `train_ofa_net.py` now uses `torchrun`, and that legacy multi-node `mpirun`/`horovodrun` commands no longer apply |
| **Requirement** | Updated from `Python 3.6+, PyTorch 1.4.0+, Horovod` to `Python 3.12+, PyTorch 2.9.1+, torchvision 0.24.1+, tqdm, filelock, gdown, pillow`; explicit note that Horovod is no longer required |

---

## Summary checklist for migrating a downstream OFA-based repo

When applying these changes to a codebase like ComPoFA that imports from OFA:

- [ ] Drop `horovod` from all `requirements.txt` / `setup.py` / conda envs
- [ ] Replace `from torch._six import string_classes` → `string_classes = (str,)`
- [ ] Replace `from torch.multiprocessing import Queue as queue` → `import queue`
- [ ] Add try/except fallback for `ExceptionWrapper` import location
- [ ] Replace PIL interpolation constants (`Image.BILINEAR` etc.) with `InterpolationMode.*`
- [ ] Add `"cuda"` as the 5th argument to `_pin_memory_loop` call
- [ ] Add `IS_WINDOWS = sys.platform == "win32"` where needed
- [ ] Replace all `hvd.allreduce` → `dist.all_reduce` (with `is_initialized()` guard)
- [ ] Replace `hvd.broadcast*` → `dist.broadcast_object_list` + `dist.broadcast`
- [ ] Replace `hvd.DistributedOptimizer` → manual gradient all-reduce wrapper
- [ ] Add `weights_only=False` to every `torch.load(...)` call
- [ ] Fix `nn.LogSoftmax()` → `nn.LogSoftmax(dim=1)`
- [ ] Add `_unwrap_net()` helper to handle both `DataParallel` and `DistributedDataParallel`
- [ ] Guard teacher-model KD loss with `and args.teacher_model is not None`
- [ ] Replace `mpirun`/`horovodrun` launch commands in scripts with `torchrun --nproc_per_node=N`
- [ ] Update version string in `setup.py` to use `.dev` prefix (PEP 440)
