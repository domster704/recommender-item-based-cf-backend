from src.domain.entities.movie_lens.raitings import Rating
from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.exceptions.repository import RepositoryError


class RatingCreateUseCase:
    def __init__(self, rating_repository: RepositoryInterface[Rating]):
        self.rating_repository = rating_repository

    async def execute(self, rating: Rating) -> Rating | None:
        try:
            return await self.rating_repository.add(rating)
        except RepositoryError as e:
            return None
