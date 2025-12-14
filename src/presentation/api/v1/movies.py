from fastapi import APIRouter, Depends

from src.application.usecase.movies.get import MoviesGetUseCase
from src.domain.entities.movie_lens.movie import Movie
from src.presentation.dependencies.movies.get import get_movies_use_case

movies_router = APIRouter(prefix="/movies", tags=["movies"])


@movies_router.get("/")
async def get_all_movies(
    use_case: MoviesGetUseCase = Depends(get_movies_use_case),
) -> list[Movie]:
    return await use_case.get_popular()


@movies_router.get("/{movie_id}")
async def get_movie_by_ud(
    movie_id: int,
    use_case: MoviesGetUseCase = Depends(get_movies_use_case),
) -> Movie | None:
    return await use_case.get_by_id(movie_id)
