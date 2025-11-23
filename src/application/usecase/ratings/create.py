from src.domain.entities.movie_lens.movie import Movie
from src.domain.entities.movie_lens.raitings import Rating
from src.domain.entities.movie_lens.user import User
from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.exceptions.repository import RepositoryError


class RatingCreateUseCase:
    def __init__(
            self,
            user_repository: RepositoryInterface[User],
            movie_repository: RepositoryInterface[Movie],
            rating_repository: RepositoryInterface[Rating]
    ):
        self.rating_repository = rating_repository
        self.movie_repository = movie_repository
        self.user_repository = user_repository

    async def execute(self, movie_id: int, tg_user_id: int, rating: int) -> Rating | None:
        try:
            user: User | None = await self.user_repository.get(tg_user_id, field_search="tg_user_id")
            movie: Movie | None = await self.movie_repository.get(movie_id, field_search="id")

            if user is None or movie is None:
                raise RepositoryError("User or movie not found")

            return await self.rating_repository.add(Rating(
                user=user,
                movie=movie,
                rating=rating,
                timestamp=None
            ))
        except RepositoryError as e:
            return None
