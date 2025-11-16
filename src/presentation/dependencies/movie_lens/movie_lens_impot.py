from src.application.usecase.movie_lens.movie_lens_import import MovieLensImportUseCase


def get_movie_lens_import_use_case() -> MovieLensImportUseCase:
    return MovieLensImportUseCase()
