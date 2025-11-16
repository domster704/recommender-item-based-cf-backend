from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from src.infrastructure.config.settings import settings

DB_URL = settings.db.data_source_name

engine = create_async_engine(DB_URL)
AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=True,
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
