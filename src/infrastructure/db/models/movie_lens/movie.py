from __future__ import annotations

from datetime import date

from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship

from src.domain.entities.movie_lens.movie import Movie
from src.infrastructure.db.models import BaseORM
from src.infrastructure.db.models.movie_lens.links import MovieGenreLink


class MovieORM(BaseORM, table=True):
    __tablename__ = "movie"

    id: int | None = Field(default=None, primary_key=True)
    title: str
    release_date: date | None
    video_release_date: date | None
    imdb_url: str

    genres: list["GenreORM"] = Relationship(
        sa_relationship=relationship(
            "GenreORM",
            secondary=MovieGenreLink.__table__,
            back_populates="movies",
            passive_deletes=True,
            lazy="selectin",
        )
    )

    ratings: list["RatingORM"] = Relationship(
        sa_relationship=relationship(
            "RatingORM",
            back_populates="movie",
            passive_deletes=True,
            # lazy="selectin"
        )
    )

    def to_entity(self) -> Movie:
        return Movie(
            id=self.id,
            title=self.title,
            release_date=self.release_date,
            video_release_date=self.video_release_date,
            imdb_url=self.imdb_url,
            genres=[genre.to_entity() for genre in self.genres],
        )

    @classmethod
    def from_entity(cls, entity: Movie) -> MovieORM:
        return cls(
            id=entity.id,
            title=entity.title,
            release_date=entity.release_date,
            video_release_date=entity.video_release_date,
            imdb_url=entity.imdb_url,
        )
