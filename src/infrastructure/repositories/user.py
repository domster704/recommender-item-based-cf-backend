from src.domain.entities.movie_lens.user import User
from src.infrastructure.db.models import UserORM
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.base import BaseRepository


class UserRepository(BaseRepository[UserORM, User]):
    model = UserORM
    entity = User

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow=uow)
