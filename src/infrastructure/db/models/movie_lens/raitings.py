from __future__ import annotations

from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship

from src.domain.entities.movie_lens.raitings import Rating
from src.infrastructure.db.models import BaseORM


class RatingORM(BaseORM, table=True):
    __tablename__ = "rating"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    movie_id: int = Field(foreign_key="movie.id")
    rating: int
    timestamp: int

    user: "UserORM" = Relationship(
        sa_relationship=relationship(
            "UserORM", back_populates="ratings", lazy="selectin"
        )
    )

    movie: "MovieORM" = Relationship(
        sa_relationship=relationship(
            "MovieORM", back_populates="ratings", lazy="selectin"
        )
    )

    def to_entity(self) -> Rating:
        return Rating(
            user=self.user.to_entity(),
            movie=self.movie.to_entity(),
            rating=self.rating,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_entity(cls, entity: Rating) -> RatingORM:
        return cls(
            user_id=entity.user.id,
            movie_id=entity.movie.id,
            rating=entity.rating,
            timestamp=entity.timestamp,
        )
