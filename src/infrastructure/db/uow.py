from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession

from src.infrastructure.db.session import AsyncSessionLocal


class UnitOfWork:
    """Класс для управления единицей работы (Unit of Work) с базой данных.

    Этот класс предоставляет контекстный менеджер для работы с асинхронной сессией SQLAlchemy.
    Он автоматически управляет открытием и закрытием сессии, а также обработкой транзакций.

    Attributes:
        session (AsyncSession): Асинхронная сессия SQLAlchemy
    """

    def __init__(self):
        """Инициализирует экземпляр UnitOfWork."""
        self.session: AsyncSession = None

    async def __aenter__(self):
        """Открывает асинхронную сессию при входе в контекстный менеджер.

        Returns:
            UnitOfWork: Текущий экземпляр UnitOfWork
        """
        self.session = AsyncSessionLocal()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрывает асинхронную сессию при выходе из контекстного менеджера.

        Args:
            exc_type: Тип исключения, если оно произошло
            exc_val: Значение исключения, если оно произошло
            exc_tb: Трассировка стека исключения, если оно произошло
        """
        if self.session:
            await self.session.close()

    async def commit(self) -> bool:
        """Фиксирует изменения в базе данных.

        Returns:
            bool: True, если коммит выполнен успешно, иначе False.

        Raises:
            SQLAlchemyError: Если произошла ошибка при выполнении коммита.
        """
        try:
            await self.session.commit()
            return True
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise
