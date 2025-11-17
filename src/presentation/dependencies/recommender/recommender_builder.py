from pathlib import Path

from src.application.providers.uow import uow_context
from src.application.usecase.recommender.recommender_builder import (
    RecommenderBuilderUseCase,
)
from src.infrastructure.repositories.movie import MovieRepository
from src.infrastructure.repositories.rating import RatingRepository
from src.infrastructure.services.recommender_module.recommender_service import (
    RecommenderService,
)
from src.infrastructure.services.similarity_cache import PickleSimilarityCache

BASE_DIR = Path(__file__).resolve().parents[3]


async def recommender_builder() -> RecommenderBuilderUseCase:
    async with uow_context() as uow:
        rating_repository = RatingRepository(uow)
        movie_repository = MovieRepository(uow)

    CACHE_PATH = BASE_DIR / "shared" / "assets" / "similarity.pkl"
    cache = PickleSimilarityCache(path=CACHE_PATH)

    return RecommenderBuilderUseCase(
        rating_repository=rating_repository,
        movie_repository=movie_repository,
        recommender=RecommenderService(cache=cache),
    )
