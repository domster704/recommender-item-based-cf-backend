from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from sqlmodel import SQLModel

EntityType = TypeVar("EntityType")


class Base(SQLModel, table=False): ...


class BaseORM(Base, Generic[EntityType], ABC):
    """Абстрактная база для ORM-моделей."""

    @abstractmethod
    def to_entity(self) -> EntityType:
        """Преобразует ORM-модель в доменную сущность."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_entity(cls, entity: EntityType) -> Base:
        """Преобразует доменную сущность в ORM-модель."""
        raise NotImplementedError
