import os
from argparse import Namespace
from typing import Any

from src.sv.sv_main import SVBase

BASE_DIR = os.getenv("BASE_COE")


class ProbeBase(SVBase):
    """Base class for layer-wise probing analyzers."""

    @staticmethod
    def _output_dir(args: Namespace) -> str:
        subdir = "probe_ood" if bool(args.ood) else "probe_id"
        out_dir = os.path.join(BASE_DIR, "output", subdir, f"sandbox_{args.mode}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def run(self, args: Namespace) -> dict[str, Any]:
        raise NotImplementedError("ProbeBase.run must be implemented in subclasses.")
