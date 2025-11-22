from dataclasses import dataclass
from enum import StrEnum

from src.domain.entities.movie_lens.occupation import Occupation


class UserGender(StrEnum):
    M = "male"
    F = "female"


@dataclass(frozen=True, slots=True)
class User:
    id: int | None
    age: int
    gender: UserGender
    occupation: Occupation | None
    tg_user_id: int | None = None
