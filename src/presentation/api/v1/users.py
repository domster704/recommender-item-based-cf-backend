from fastapi import APIRouter, Depends

from src.application.usecase.users.create import UserCreateUseCase
from src.application.usecase.users.get import UsersGetUseCase
from src.domain.entities.movie_lens.user import User
from src.presentation.dependencies.users.create import get_user_create_use_case
from src.presentation.dependencies.users.get import get_user_get_use_case
from src.presentation.dependencies.users.update import get_user_update_use_case
from src.presentation.schemas.users.create import UserCreateBody

users_router = APIRouter(prefix="/users", tags=["users"])


@users_router.get("/")
async def get_all_users(
    use_case: UsersGetUseCase = Depends(get_user_get_use_case),
):
    return await use_case.get_all()


@users_router.get("/{tg_user_id}")
async def get_user(
    tg_user_id: str,
    use_case: UsersGetUseCase = Depends(get_user_get_use_case),
):
    return await use_case.get_by_id(int(tg_user_id))


@users_router.post("/")
async def get_user_by_tg_user_id(
    user: UserCreateBody,
    use_case: UserCreateUseCase = Depends(get_user_create_use_case),
):
    return await use_case.execute(
        User(
            id=None,
            age=user.age,
            gender=user.gender,
            occupation=None,
            tg_user_id=user.tg_user_id,
        )
    )


@users_router.put("/")
async def update_user(
    user: User,
    use_case: UserCreateUseCase = Depends(get_user_update_use_case),
):
    return await use_case.execute(user)
