from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import JSONResponse

from src.application.usecase.movies.get import MoviesGetUseCase
from src.application.usecase.users.get import UsersGetUseCase
from src.domain.entities.movie_lens.user import User
from src.domain.interfaces.recommender import IRecommender
from src.presentation.api.v1.users import get_all_users
from src.presentation.dependencies.movies.get import get_movies_use_case
from src.presentation.dependencies.recommender.get_recommender import get_recommender
from src.presentation.dependencies.users.get import get_user_get_use_case

recommendations_router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@recommendations_router.get("/{tg_user_id}")
async def get_recommendations(
    tg_user_id: int,
    top_n: int = 10,
    recommender: IRecommender = Depends(get_recommender),
    movie_use_case: MoviesGetUseCase = Depends(get_movies_use_case),
    user_use_case: UsersGetUseCase = Depends(get_user_get_use_case),
):
    user: User | None = await user_use_case.get_by_tg_user_id(tg_user_id)
    if not user:
        return JSONResponse(status_code=404, content={"message": "User not found"})

    movie_ids: list[int] = await recommender.recommend_for_user(user.id, top_n)
    return await movie_use_case.get_by_ids(movie_ids)
