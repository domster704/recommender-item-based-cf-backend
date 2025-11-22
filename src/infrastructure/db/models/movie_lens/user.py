from __future__ import annotations

from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship

from src.domain.entities.movie_lens.user import UserGender, User
from src.infrastructure.db.models import BaseORM


class UserORM(BaseORM, table=True):
    __tablename__ = "user"

    id: int | None = Field(default=None, primary_key=True)
    age: int
    gender: UserGender = Field(max_length=1)
    occupation_id: int | None = Field(foreign_key="occupation.id", nullable=True)
    tg_user_id: int | None = Field(unique=True, nullable=True)

    occupation: "OccupationORM | None" = Relationship(
        sa_relationship=relationship(
            "OccupationORM", back_populates="users", lazy="selectin"
        )
    )

    ratings: list["RatingORM"] = Relationship(
        sa_relationship=relationship(
            "RatingORM",
            back_populates="user",
            passive_deletes=True,
            # lazy="selectin"
        )
    )

    def to_entity(self) -> User:
        return User(
            id=self.id,
            age=self.age,
            gender=self.gender,
            occupation=self.occupation.to_entity() if self.occupation else None,
        )

    @classmethod
    def from_entity(cls, entity: User) -> UserORM:
        return cls(
            id=entity.id,
            age=entity.age,
            gender=entity.gender,
            occupation_id=entity.occupation.id if entity.occupation else None,
        )
