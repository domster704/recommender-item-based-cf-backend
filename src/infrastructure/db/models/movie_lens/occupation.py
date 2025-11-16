from __future__ import annotations

from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship

from src.domain.entities.movie_lens.occupation import Occupation
from src.infrastructure.db.models import BaseORM
from src.infrastructure.db.models.movie_lens.user import UserORM


class OccupationORM(BaseORM, table=True):
    __tablename__ = "occupation"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)

    users: list[UserORM] = Relationship(
        sa_relationship=relationship(
            "UserORM", back_populates="occupation", lazy="selectin"
        )
    )

    def to_entity(self) -> Occupation:
        return Occupation(id=self.id, name=self.name)

    @classmethod
    def from_entity(cls, entity: Occupation) -> OccupationORM:
        return cls(id=entity.id, name=entity.name)
