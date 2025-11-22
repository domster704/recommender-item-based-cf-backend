from src.domain.entities.movie_lens.occupation import Occupation
from src.domain.entities.movie_lens.user import User
from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.exceptions.repository import RepositoryError


class OccupationsGetUseCase:
    def __init__(self, occupation_repository: RepositoryInterface[Occupation]):
        self.occupation_repository = occupation_repository

    async def get_all(self) -> list[Occupation]:
        movies: list[Occupation] = await self.occupation_repository.get_all()
        return movies

    async def get_by_id(self, occupation_id: int) -> Occupation | None:
        try:
            return await self.occupation_repository.get(
                occupation_id, field_search="id"
            )
        except RepositoryError as e:
            return None
