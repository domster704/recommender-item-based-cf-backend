from fastapi.params import Depends

from src.application.providers.uow import uow_provider
from src.application.usecase.users.update import UserUpdateUseCase
from src.infrastructure.repositories.user import UserRepository


def get_user_update_use_case(uow=Depends(uow_provider)) -> UserUpdateUseCase:
    return UserUpdateUseCase(user_repository=UserRepository(uow))
