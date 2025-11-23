from fastapi import APIRouter, Depends

from src.application.usecase.movies.get import MoviesGetUseCase
from src.presentation.dependencies.movies.get import get_movies_use_case

movies_router = APIRouter(prefix="/movies", tags=["movies"])


@movies_router.get("/")
async def get_all_movies(
    use_case: MoviesGetUseCase = Depends(get_movies_use_case),
):
    return await use_case.get_all()

@movies_router.get("/{movie_id}")
async def get_movie_by_ud(
        movie_id: int,
        use_case: MoviesGetUseCase = Depends(get_movies_use_case),
):
    return await use_case.get_by_id(movie_id)
