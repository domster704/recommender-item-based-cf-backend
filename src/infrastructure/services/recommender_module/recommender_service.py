from typing import Callable, Awaitable

from src.domain.entities.movie_lens.movie import Movie
from src.domain.entities.movie_lens.raitings import Rating
from src.domain.interfaces.recommender import IRecommenderBuilder, IRecommender
from src.domain.interfaces.similarity_cache import ISimilarityCache
from src.infrastructure.services.recommender_module.recommender.item_based_cf_recommender import (
    ItemBasedCFRecommender,
)
from src.infrastructure.services.recommender_module.similarity.builder import (
    SimilarityMatrixBuilder,
)
from src.infrastructure.services.recommender_module.storage.ratings_storage import (
    RatingsStorage,
)


class RecommenderService(IRecommenderBuilder):
    """
    Сервис, собирающий весь pipeline:
    - загрузка данных
    - кэширование
    - построение матрицы
    - создание рекомендателя
    """

    def __init__(self, cache: ISimilarityCache | None = None):
        self.cache = cache

    async def build(
        self,
        ratings_loader: Callable[[], Awaitable[list[Rating]]],
        movies_loader: Callable[[], Awaitable[list[Movie]]],
    ) -> IRecommender:
        storage = RatingsStorage()
        sim_matrix = None

        if self.cache:
            state: dict[int, dict[int, float]] | None = await self.cache.load()
            if state:
                storage.set_users(state["user_ratings"])
                sim_matrix = state["similarity_matrix"]

        if sim_matrix is None:
            ratings: list[Rating] = await ratings_loader()
            movies: list[Movie] = await movies_loader()

            storage.fill(ratings)
            builder = SimilarityMatrixBuilder()
            sim_matrix: dict[int, dict[int, float]] = builder.build(
                storage.users, movies
            )

            if self.cache:
                await self.cache.save(
                    {"user_ratings": storage.users, "similarity_matrix": sim_matrix}
                )

        return ItemBasedCFRecommender(sim_matrix, storage, self.cache)
