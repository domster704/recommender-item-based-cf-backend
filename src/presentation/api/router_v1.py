from fastapi import APIRouter

from src.presentation.api.v1.recommendations import recommendations_router
from src.presentation.api.v1.movies import movies_router
from src.presentation.api.v1.occupations import occupations_router
from src.presentation.api.v1.users import users_router


api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(recommendations_router)
api_v1_router.include_router(movies_router)
api_v1_router.include_router(occupations_router)
api_v1_router.include_router(users_router)
