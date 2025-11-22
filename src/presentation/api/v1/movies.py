from fastapi import APIRouter, Depends

from src.application.usecase.movies.get_all import MoviesGetAllUseCase
from src.presentation.dependencies.movies.get_all import get_all_movies_use_case

movies_router = APIRouter(prefix="/movies", tags=["movies"])


@movies_router.get("/")
async def get_all_movies(
    use_case: MoviesGetAllUseCase = Depends(get_all_movies_use_case),
):
    return await use_case.execute()
