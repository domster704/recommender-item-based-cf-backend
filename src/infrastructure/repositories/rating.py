from src.domain.entities.movie_lens.raitings import Rating
from src.infrastructure.db.models import RatingORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.base import BaseRepository


class RatingRepository(BaseRepository[RatingORM, Rating]):
    model = RatingORM
    entity = Rating

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow=uow)
