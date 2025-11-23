from src.domain.entities.movie_lens.movie import Movie
from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.exceptions.repository import RepositoryError


class MoviesGetUseCase:
    def __init__(self, movie_repository: RepositoryInterface[Movie]):
        self.movie_repository = movie_repository

    async def get_all(self) -> list[Movie]:
        movies: list[Movie] = await self.movie_repository.get_all()
        return movies

    async def get_by_id(self, movie_id: int) -> Movie | None:
        try:
            return await self.movie_repository.get(movie_id, field_search="id")
        except RepositoryError as e:
            return None