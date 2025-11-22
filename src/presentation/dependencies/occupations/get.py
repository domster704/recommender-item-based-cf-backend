from fastapi import Depends

from src.application.providers.uow import uow_provider
from src.application.usecase.occupations.get import OccupationsGetUseCase
from src.infrastructure.repositories.occupation import OccupationRepository


def get_occupation_get_use_case(uow=Depends(uow_provider)) -> OccupationsGetUseCase:
    return OccupationsGetUseCase(occupation_repository=OccupationRepository(uow))
