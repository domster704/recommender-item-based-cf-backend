from src.domain.entities.movie_lens.movie import Movie
from src.domain.entities.movie_lens.raitings import Rating
from src.domain.entities.movie_lens.user import User
from src.domain.repositories.base import RepositoryInterface
from src.infrastructure.exceptions.repository import RepositoryError
from src.infrastructure.repositories.rating import RatingRepository


class RatingCreateUseCase:
    def __init__(
            self,
            user_repository: RepositoryInterface[User],
            movie_repository: RepositoryInterface[Movie],
            rating_repository: RatingRepository
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

            existing = await self.rating_repository.get_by_user_and_movie(
                user_id=user.id,
                movie_id=movie.id
            )

            if existing:
                new_rating = Rating(
                    id=existing.id,
                    user=existing.user,
                    movie=existing.movie,
                    rating=rating,
                    timestamp=None
                )
                return await self.rating_repository.update(new_rating)

            return await self.rating_repository.add(Rating(
                user=user,
                movie=movie,
                rating=rating,
                timestamp=None
            ))
        except RepositoryError as e:
            return None
