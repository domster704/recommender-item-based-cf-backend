from src.infrastructure.db.models import Base
from src.infrastructure.db.session import engine


async def init_db():
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    import asyncio

    asyncio.run(init_db())
