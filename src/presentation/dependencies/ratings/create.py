from fastapi import Depends

from src.application.providers.uow import uow_provider
from src.application.usecase.ratings.create import RatingCreateUseCase
from src.infrastructure.repositories.movie import MovieRepository
from src.infrastructure.repositories.rating import RatingRepository
from src.infrastructure.repositories.user import UserRepository


def get_rating_create_use_case(uow=Depends(uow_provider)) -> RatingCreateUseCase:
    return RatingCreateUseCase(
        rating_repository=RatingRepository(uow),
        movie_repository=MovieRepository(uow),
        user_repository=UserRepository(uow)
    )
