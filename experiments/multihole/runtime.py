"""Shared portable resource guard, loaded without CUDA until evaluation."""
from pathlib import Path
import importlib.util
path=Path(__file__).resolve().parents[1]/'periodic/runtime.py'
spec=importlib.util.spec_from_file_location('neb2d_resource',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
guard=module.guard
