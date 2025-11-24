from pathlib import Path

from src.infrastructure.services.movie_lens_importer import MovieLensImporter


class MovieLensImportUseCase:
    def __init__(self):
        pass

    async def execute(self, movie_lens_path: Path) -> None:
        service = MovieLensImporter(movie_lens_path)
        await service.import_all()
