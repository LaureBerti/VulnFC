#!/usr/bin/env python3
"""Register a fact-checking system as a victim so the benchmark can attack it.

A victim is any object exposing one method:

    predict(claim: str, shot_mode: str) -> int | None

returning 1 for TRUE, 0 for FAKE, or None when the system produced no parseable
verdict. None is counted as unavailable and reported; it is never silently dropped,
because a system that cannot answer is not a system that proved robust.

Everything else in the benchmark -- the attack taxonomy, the budgets, the
learnability gate and the reported metrics -- applies unchanged once a system
satisfies that one method.

Two ways to register:

1. In-tree, with the decorator:

       from victim_api import register_victim

       @register_victim("my_system")
       class MySystem:
           def __init__(self, spec: str = ""):
               ...
           def predict(self, claim, shot_mode):
               return 1 if ... else 0

   then run with  --victim_model my_system
   or             --victim_model my_system:<free-form spec>

2. Out-of-tree, by import path, with no edit to this repository:

       --victim_model py:mypackage.victims:MySystem
       --victim_model py:mypackage.victims:MySystem:<free-form spec>

The <spec> is passed to the constructor verbatim; use it for an endpoint, a
checkpoint path or a config name.

Trust boundary: an import path is read from this process's own command line and
is therefore operator-supplied. Never pass a module path that came from a
dataset, a claim, a model response or any file the benchmark downloads, since
importing a module executes it.
"""
from __future__ import annotations

import importlib
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Victim(Protocol):
    """The one method the benchmark requires of a fact-checking system."""

    def predict(self, claim: str, shot_mode: str) -> Optional[int]:
        ...


VICTIM_REGISTRY: dict[str, type] = {}

PLUGIN_PREFIX = "py:"


def register_victim(name: str) -> Callable[[type], type]:
    """Register a victim class under `name` for use with --victim_model."""
    def wrap(cls: type) -> type:
        if name in VICTIM_REGISTRY and VICTIM_REGISTRY[name] is not cls:
            raise ValueError(f"victim name already registered: {name!r}")
        VICTIM_REGISTRY[name] = cls
        return cls
    return wrap


def is_victim_spec(spec: str) -> bool:
    """True when `spec` names a registered or importable victim."""
    if spec.startswith(PLUGIN_PREFIX):
        return True
    return spec.split(":", 1)[0] in VICTIM_REGISTRY


def _check(obj: object, origin: str) -> Victim:
    if not callable(getattr(obj, "predict", None)):
        raise TypeError(f"{origin} has no callable predict(claim, shot_mode)")
    return obj


def load_victim(spec: str) -> Victim:
    """Build a victim from a registry name or a py: import path.

    Raises ValueError or TypeError rather than returning a victim that cannot be
    queried, so a misconfigured run fails at startup instead of recording every
    claim as unavailable.
    """
    if spec.startswith(PLUGIN_PREFIX):
        rest = spec[len(PLUGIN_PREFIX):]
        parts = rest.split(":", 2)
        if len(parts) < 2:
            raise ValueError(
                f"expected {PLUGIN_PREFIX}module.path:ClassName[:spec], got {spec!r}")
        module_path, class_name = parts[0], parts[1]
        arg = parts[2] if len(parts) == 3 else ""
        module = importlib.import_module(module_path)
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(f"{module_path} defines no {class_name}") from exc
        return _check(cls(arg) if arg else cls(), f"{module_path}:{class_name}")

    name, _, arg = spec.partition(":")
    if name not in VICTIM_REGISTRY:
        raise ValueError(
            f"unknown victim {name!r}; registered: {sorted(VICTIM_REGISTRY)}")
    cls = VICTIM_REGISTRY[name]
    return _check(cls(arg) if arg else cls(), name)


def available() -> list[str]:
    """Registered victim names, for --help text and error messages."""
    return sorted(VICTIM_REGISTRY)
