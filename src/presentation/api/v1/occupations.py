from fastapi import APIRouter, Depends

from src.application.usecase.ratings.create import RatingCreateUseCase
from src.domain.entities.movie_lens.raitings import Rating
from src.presentation.dependencies.ratings.create import get_rating_create_use_case

occupations_router = APIRouter(prefix="/occupations", tags=["occupations"])


@occupations_router.get("/")
async def get_all_occupations(
    rating: Rating,
    use_case: RatingCreateUseCase = Depends(get_rating_create_use_case),
):
    return await use_case.execute(rating)
