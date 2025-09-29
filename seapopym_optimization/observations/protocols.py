"""Protocols for SeapoPym optimization algorithms and components."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ObservationProtocol(Protocol):
    """Protocol for observations used in cost functions."""

    name: str
    observation: object
