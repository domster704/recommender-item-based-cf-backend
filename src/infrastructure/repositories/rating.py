from sqlmodel import select

from src.domain.entities.movie_lens.raitings import Rating
from src.infrastructure.db.models import RatingORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.base import BaseRepository


class RatingRepository(BaseRepository[RatingORM, Rating]):
    model = RatingORM
    entity = Rating

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow=uow)

    async def get_by_user_and_movie(self, user_id: int, movie_id: int) -> Rating | None:
        stmt = (
            select(RatingORM)
            .where(RatingORM.user_id == user_id)
            .where(RatingORM.movie_id == movie_id)
        )
        result = await self.uow.session.exec(stmt)
        model = result.first()
        return model.to_entity() if model else None
