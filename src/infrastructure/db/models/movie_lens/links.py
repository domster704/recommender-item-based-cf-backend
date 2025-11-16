from sqlmodel import Field

from src.infrastructure.db.models import Base


class MovieGenreLink(Base, table=True):
    movie_id: int = Field(foreign_key="movie.id", primary_key=True)
    genre_id: int = Field(foreign_key="genre.id", primary_key=True)
