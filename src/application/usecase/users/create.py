from src.domain.entities.movie_lens.user import User
from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.exceptions.repository import RepositoryError


class UserCreateUseCase:
    def __init__(self, user_repository: RepositoryInterface[User]):
        self.user_repository = user_repository

    async def execute(self, user: User) -> User | None:
        try:
            return await self.user_repository.add(user)
        except RepositoryError as e:
            print(e)
            return None
