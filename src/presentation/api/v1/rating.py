from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.application.usecase.ratings.create import RatingCreateUseCase
from src.application.usecase.recommender.update_recommender import (
    UpdateRecommenderUseCase,
)
from src.domain.entities.movie_lens.raitings import Rating
from src.domain.interfaces.recommender import IRecommender
from src.presentation.dependencies.ratings.create import get_rating_create_use_case
from src.presentation.dependencies.recommender.update_recommender import (
    get_update_recommender_use_case,
)

ratings_router = APIRouter(prefix="/ratings", tags=["ratings"])


class RatingsCreateSchema(BaseModel):
    tg_user_id: int
    movie_id: int
    rating: int


@ratings_router.post("/")
async def add_rating(
    rating: RatingsCreateSchema,
    use_case: RatingCreateUseCase = Depends(get_rating_create_use_case),
    update_recommender: UpdateRecommenderUseCase = Depends(
        get_update_recommender_use_case
    ),
):
    new_rating = await use_case.execute(**rating.model_dump())

    if new_rating:
        await update_recommender.execute(new_rating)

    return new_rating
