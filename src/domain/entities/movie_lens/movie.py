from dataclasses import dataclass
from datetime import date

from src.domain.entities.movie_lens.genre import Genre


@dataclass(frozen=True, slots=True)
class Movie:
    id: int
    title: str
    release_date: date
    video_release_date: date
    imdb_url: str
    genres: list[Genre]
