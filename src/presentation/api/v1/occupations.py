from fastapi import APIRouter, Depends

from src.application.usecase.occupations.get import OccupationsGetUseCase
from src.domain.entities.movie_lens.occupation import Occupation
from src.presentation.dependencies.occupations.get import get_occupation_get_use_case

occupations_router = APIRouter(prefix="/occupations", tags=["occupations"])


@occupations_router.get("/")
async def get_all_occupations(
    use_case: OccupationsGetUseCase = Depends(get_occupation_get_use_case),
) -> list[Occupation]:
    return await use_case.get_all()
