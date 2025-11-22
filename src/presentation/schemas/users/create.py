from pydantic import BaseModel

from src.domain.entities.movie_lens.user import UserGender


class UserCreateBody(BaseModel):
    tg_user_id: int
    age: int
    gender: UserGender
