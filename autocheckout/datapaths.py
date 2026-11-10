"""The one data folder of the GPU runs (README.md inside it): next to the repo as ``../autocheckout-data``, or
wherever ``AC_DATA`` points.

    autocheckout-data/
    ├── rpc/{val2019,test2019}       real RPC checkout images
    ├── eval/                        val/test labels, task_config.json (rpc_eval_official_v1)
    ├── synth_v2/                    synthetic training release, tasks 1-5 (rpc_synth_v2_v1)
    ├── synth_v2_12x/                tasks 1-9 of the 12+13-SKU split (rpc_synth_v2_12x_v1; task 1 = synth_v2's)
    ├── eval_12x/                    val/test labels of the 12+13-SKU split (same images and boxes as eval/)
    ├── pretrained/hf-cache/         Deformable DETR + ConvNeXt-V2 weights
    ├── models/full-syn-r1/          FULL-SYN round 1: FSA, tasks 1-5, counting policies, metrics
    └── relearn/                     relearning + active learning: meta/, variants/, active/, images/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAME = 'autocheckout-data'


def data_root() -> Path:
    return Path(os.environ.get('AC_DATA') or REPO.parent / NAME).expanduser().resolve()


@dataclass(frozen=True)
class Layout:
    root: Path

    @classmethod
    def at(cls, root: str | os.PathLike | None = None) -> Layout:
        return cls(Path(root).expanduser().resolve() if root else data_root())

    @property
    def rpc(self) -> Path:
        return self.root / 'rpc'

    @property
    def eval(self) -> Path:
        return self.root / 'eval'

    @property
    def synth(self) -> Path:
        return self.root / 'synth_v2'

    @property
    def synth_12x(self) -> Path:
        return self.root / 'synth_v2_12x'

    @property
    def eval_12x(self) -> Path:
        return self.root / 'eval_12x'

    @property
    def pretrained(self) -> Path:
        return self.root / 'pretrained/hf-cache'

    @property
    def r1(self) -> Path:
        return self.root / 'models/full-syn-r1'

    @property
    def r1_task5_pred(self) -> Path:
        return self.r1 / 'runs/EXP-FULL-SYN/task_5/pred_val.npz'

    @property
    def relearn(self) -> Path:
        return self.root / 'relearn'
