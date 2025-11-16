from dataclasses import dataclass

from src.domain.entities.movie_lens.movie import Movie
from src.domain.entities.movie_lens.user import User


@dataclass(frozen=True, slots=True)
class Rating:
    user: User
    movie: Movie
    rating: int
    timestamp: int
