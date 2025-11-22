from fastapi.params import Depends

from src.application.providers.uow import uow_provider
from src.application.usecase.users.create import UserCreateUseCase
from src.infrastructure.repositories.user import UserRepository


def get_user_create_use_case(uow=Depends(uow_provider)) -> UserCreateUseCase:
    return UserCreateUseCase(user_repository=UserRepository(uow))
