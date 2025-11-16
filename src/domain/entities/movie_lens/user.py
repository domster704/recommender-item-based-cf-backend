from dataclasses import dataclass
from enum import StrEnum

from src.domain.entities.movie_lens.occupation import Occupation


class UserGender(StrEnum):
    M = "male"
    F = "female"


@dataclass(frozen=True, slots=True)
class User:
    id: int
    age: int
    gender: UserGender
    occupation: Occupation
