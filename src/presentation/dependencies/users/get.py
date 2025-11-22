from fastapi.params import Depends

from src.application.providers.uow import uow_provider
from src.application.usecase.users.get import UsersGetUseCase
from src.infrastructure.repositories.user import UserRepository


def get_user_get_use_case(uow=Depends(uow_provider)) -> UsersGetUseCase:
    return UsersGetUseCase(user_repository=UserRepository(uow))
