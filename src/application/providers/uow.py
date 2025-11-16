from contextlib import asynccontextmanager
from typing import AsyncGenerator

from src.infrastructure.db.uow import UnitOfWork


async def uow_provider() -> AsyncGenerator[UnitOfWork, None]:
    """
    Создаёт UnitOfWork для одного запроса.

    Returns:
        UnitOfWork: UnitOfWork для одного запроса
    """
    async with UnitOfWork() as uow:
        yield uow


@asynccontextmanager
async def uow_context():
    """Используется вручную, через async with"""
    async with UnitOfWork() as uow:
        yield uow
