import inspect
import logging
from typing import (
    Literal,
    Type,
    get_origin,
    get_args,
    Union,
    TypeVar,
    Awaitable,
    Any,
    Iterable,
    List,
)

from src.types.exceptions import TypeAssertionException

logger = logging.getLogger(__name__)


def _is_same_type(a, b):
    return repr(a) == repr(b)


def ensure_isimplementation(
    target: Union[Type, List[Type]],
    *bases,
    on_err: Literal["raise", "warning"] = "raise",
) -> None:
    _is = True

    bases = bases if isinstance(bases, Iterable) else [bases]

    for base in bases:
        for name, base_attr in base.__dict__.items():
            if name.startswith("_") and name != "__call__":
                continue

            if not callable(base_attr):
                continue

            if not hasattr(target, name):
                logger.critical(f"{target.__name__} missing the attribute '{name}'")
                _is = False
                break

            target_attr = getattr(target, name)

            base_sig = inspect.signature(base_attr)
            target_sig = inspect.signature(target_attr)

            if len(base_sig.parameters) > len(target_sig.parameters):
                logger.critical(
                    f"{target.__name__}.{name} has fewer parameters than {base.__name__}.{name}"
                )
                _is = False

            for b, t in zip(
                base_sig.parameters.values(), target_sig.parameters.values()
            ):
                if b.name != t.name:
                    logger.critical(
                        f"param name mismatch in {name}: {b.name} != {t.name}"
                    )
                    _is = False

                if b.kind != t.kind:
                    logger.critical(f"kind mismatch in {name}: {b.kind} != {t.kind}")
                    _is = False

                if b.default is inspect._empty and t.default is not inspect._empty:
                    logger.critical(f"param {b.name} should be required")
                    _is = False

            if (
                base_sig.return_annotation is not inspect._empty
                and target_sig.return_annotation is inspect._empty
            ):
                logger.critical(f"{target.__name__}.{name} should declare return type")
                _is = False

            if not _is_return_compatible(
                base_sig.return_annotation, target_sig.return_annotation
            ):
                logger.critical(
                    f"Return type mismatch in {target.__name__}.{name}. Expected '{base_sig.return_annotation}', got '{target_sig.return_annotation}'"
                )
                _is = False

        if not _is:
            if on_err == "raise":
                raise TypeAssertionException(target, base)

            elif on_err == "warning":
                logging.warning(
                    f"{target.__name__} is not a implementation of {base.__name__} class."
                )


def _is_return_compatible(base_ann, target_ann) -> bool:
    # Base does not specify a return annotation → ok
    if base_ann is inspect._empty:
        return True

    # Target must specify type if base does
    if target_ann is inspect._empty:
        return False

    # Handle "Any"
    if base_ann is Any:
        return True

    # Extract origins
    base_origin = get_origin(base_ann)
    target_origin = get_origin(target_ann)
    base_args = get_args(base_ann)
    target_args = get_args(target_ann)

    # OPTIONAL / NONE HANDLING
    # base: Optional[T] (Union[T, NoneType])
    # target: Optional[T]
    if base_origin is Union and type(None) in base_args:
        base_inner = [a for a in base_args if a is not type(None)][0]

        # Case 1: target is Optional[T]
        if target_origin is Union and type(None) in target_args:
            target_inner = [a for a in target_args if a is not type(None)][0]
            return _is_return_compatible(base_inner, target_inner)

        # Case 2: target is T → target can return T-only, still OK
        return _is_return_compatible(base_inner, target_ann)

    # base: T, target: Optional[T] → NOT OK (target promises more cases)
    if (
        target_origin is Union
        and type(None) in target_args
        and not (base_origin is Union and type(None) in base_args)
    ):
        return False

    # Handle non-optional Union
    if base_origin is Union:
        return any(_is_return_compatible(opt, target_ann) for opt in base_args)

    # TypeVar
    if isinstance(base_ann, TypeVar):
        return True

    # Awaitable[T]
    if base_origin is Awaitable:
        if target_origin is Awaitable:
            return True

    return _is_same_type(base_ann, target_ann)
