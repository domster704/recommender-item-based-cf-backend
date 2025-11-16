from src.domain.entities.movie_lens.genre import Genre
from src.infrastructure.db.models import GenreORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.base import BaseRepository


class GenreRepository(BaseRepository[GenreORM, Genre]):
    model = GenreORM
    entity = Genre

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow=uow)
