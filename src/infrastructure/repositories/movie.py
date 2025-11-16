from src.domain.entities.movie_lens.movie import Movie
from src.infrastructure.db.models import MovieORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.base import BaseRepository


class MovieRepository(BaseRepository[MovieORM, Movie]):
    model = MovieORM
    entity = Movie

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow=uow)
