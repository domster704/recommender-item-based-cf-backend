from sqlalchemy import func
from sqlmodel import select

from src.domain.entities.movie_lens.movie import Movie
from src.infrastructure.db.models import MovieORM, RatingORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.base import BaseRepository


class MovieRepository(BaseRepository[MovieORM, Movie]):
    model = MovieORM
    entity = Movie

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow=uow)

    async def get_all_ordered_by_popularity(self) -> list[Movie]:
        stmt = (
            select(MovieORM)
            .join(RatingORM, RatingORM.movie_id == MovieORM.id)
            .group_by(MovieORM.id)
            .order_by(func.count(RatingORM.id).desc())
        )

        result = await self.uow.session.exec(stmt)
        models = result.all()

        return [model.to_entity() for model in models]
