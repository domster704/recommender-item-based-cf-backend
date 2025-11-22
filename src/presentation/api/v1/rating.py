from fastapi import APIRouter, Depends
from src.presentation.schemas.users.update import UpdateUserSchema

from src.application.usecase.ratings.create import RatingCreateUseCase
from src.application.usecase.users.create import UserCreateUseCase
from src.application.usecase.users.get import UsersGetUseCase
from src.domain.entities.movie_lens.user import User
from src.presentation.dependencies.users.create import get_user_create_use_case
from src.presentation.dependencies.users.get import get_user_get_use_case
from src.presentation.dependencies.users.update import get_user_update_use_case
from src.presentation.schemas.users.create import UserCreateBody

ratings_router = APIRouter(prefix="/ratings", tags=["ratings"])


@ratings_router.post("/")
async def add_rating(
    user: User,
    use_case: RatingCreateUseCase = Depends(get_user_update_use_case),
):
    return await use_case.execute(user)
