from src.domain.entities.movie_lens.occupation import Occupation
from src.infrastructure.db.models import OccupationORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.base import BaseRepository


class OccupationRepository(BaseRepository[OccupationORM, Occupation]):
    model = OccupationORM
    entity = Occupation

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow=uow)
