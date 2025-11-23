from fastapi.params import Depends

from src.application.providers.uow import uow_provider
from src.application.usecase.movies.get import MoviesGetUseCase
from src.infrastructure.repositories.movie import MovieRepository


def get_movies_use_case(uow=Depends(uow_provider)) -> MoviesGetUseCase:
    return MoviesGetUseCase(movie_repository=MovieRepository(uow))
