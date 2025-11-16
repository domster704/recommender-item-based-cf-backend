from typing import Type, TypeVar, Generic

from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import select

from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.db.models import BaseORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.exceptions.repository import RepositoryError

ModelType = TypeVar("ModelType", bound=BaseORM)
EntityType = TypeVar("EntityType")


class BaseRepository(Generic[ModelType, EntityType], RepositoryInterface[EntityType]):
    """Базовый универсальный репозиторий для работы с моделями SQLModel.

    Этот класс реализует базовые CRUD-операции (создание, чтение, обновление, удаление)
    для работы с базой данных.

    Attributes:
        model (Type[ModelType]): Модель SQLModel, с которой работает репозиторий
        uow (UnitOfWork): Единица работы для управления сессией
    """

    model: type[ModelType]
    entity: type[EntityType]

    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def add(
        self, entity: EntityType, commit: bool = True, **kwargs
    ) -> EntityType:
        try:
            self.uow.session.add(self.model.from_entity(entity))

            if commit:
                await self.uow.commit()

            return entity
        except SQLAlchemyError as e:
            raise RepositoryError(e)

    async def get(self, reference: int | str, field_search: str) -> EntityType | None:
        if not hasattr(self.model, field_search):
            raise RepositoryError(
                f"Field {field_search} does not exist in model {self.model}"
            )

        result = await self.uow.session.exec(
            select(self.model).where(getattr(self.model, field_search) == reference)
        )
        model: ModelType | None = result.first()
        if not model:
            return None

        return model.to_entity()

    async def _get_model(
        self, reference: int | str, field_search: str = "id"
    ) -> ModelType | None:
        """Возвращает ORM-объект по значению поля.
        Используется внутри delete, update и других внутренних операций.
        """
        if not hasattr(self.model, field_search):
            raise RepositoryError(
                f"Field {field_search} does not exist in model {self.model}"
            )

        result = await self.uow.session.exec(
            select(self.model).where(getattr(self.model, field_search) == reference)
        )
        return result.first()

    async def get_all(self, **kwargs) -> list[EntityType]:
        result = await self.uow.session.exec(select(self.model))
        models: list[ModelType] = result.all()

        return [model.to_entity() for model in models]

    async def get_all_by_ids(self, ids: list[int] | list[str]) -> list[EntityType]:
        if not ids:
            return []

        stmt = select(self.model).where(self.model.id.in_(ids))
        result = await self.uow.session.exec(stmt)

        models: list[ModelType] = result.all()
        return [model.to_entity() for model in models]

    async def delete(self, reference: int | str) -> bool:
        model: ModelType | None = await self._get_model(reference)
        if model is None:
            return False

        await self.uow.session.delete(model)
        return await self.uow.commit()

    async def update(self, entity: EntityType) -> EntityType:
        if not entity:
            raise RepositoryError("Cannot update non-existent entity")

        model = await self._get_model(entity.id)
        if model is None:
            raise RepositoryError(f"Object with id={entity.id} not found")

        updated_data = self.model.from_entity(entity)
        await self.uow.session.merge(updated_data)
        await self.uow.session.commit()
        return entity
