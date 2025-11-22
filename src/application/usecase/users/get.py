from src.domain.entities.movie_lens.user import User
from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.exceptions.repository import RepositoryError


class UsersGetUseCase:
    def __init__(self, user_repository: RepositoryInterface[User]):
        self.user_repository = user_repository

    async def get_all(self) -> list[User]:
        movies: list[User] = await self.user_repository.get_all()
        return movies

    async def get_by_id(self, user_id: int) -> User | None:
        try:
            return await self.user_repository.get(user_id, field_search="id")
        except RepositoryError as e:
            return None
