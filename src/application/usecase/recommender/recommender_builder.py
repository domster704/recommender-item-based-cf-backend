from src.domain.entities.movie_lens.movie import Movie
from src.domain.entities.movie_lens.raitings import Rating
from src.domain.interfaces.recommender import IRecommender, IRecommenderBuilder
from src.domain.repositories.base import RepositoryInterface


class RecommenderBuilderUseCase:
    def __init__(
        self,
        rating_repository: RepositoryInterface[Rating],
        movie_repository: RepositoryInterface[Movie],
        recommender: IRecommenderBuilder,
    ):
        self.rating_repository = rating_repository
        self.movie_repository = movie_repository
        self.recommender = recommender

    async def execute(self) -> IRecommender:
        ratings = self.rating_repository.get_all
        movies = self.movie_repository.get_all

        recommender_service: IRecommender = await self.recommender.build(
            ratings, movies
        )
        return recommender_service
