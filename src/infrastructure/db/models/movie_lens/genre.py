from __future__ import annotations

from sqlalchemy.orm import relationship
from sqlmodel import Field
from sqlmodel import Relationship

from src.domain.entities.movie_lens.genre import Genre
from src.infrastructure.db.models.movie_lens.links import MovieGenreLink
from src.infrastructure.db.models import BaseORM


class GenreORM(BaseORM, table=True):
    __tablename__ = "genre"

    id: int | None = Field(default=None, primary_key=True)
    name: str

    movies: list["MovieORM"] = Relationship(
        sa_relationship=relationship(
            "MovieORM",
            secondary=MovieGenreLink.__table__,
            back_populates="genres",
            passive_deletes=True,
            lazy="selectin",
        )
    )

    def to_entity(self) -> Genre:
        return Genre(
            id=self.id,
            name=self.name,
        )

    @classmethod
    def from_entity(cls, entity: Genre) -> GenreORM:
        return cls(
            id=entity.id,
            name=entity.name,
        )
