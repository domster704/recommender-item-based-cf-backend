from fastapi import APIRouter
from fastapi.params import Depends

from src.domain.interfaces.recommender import IRecommender
from src.presentation.dependencies.recommender.get_recommender import get_recommender

recommendations_router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@recommendations_router.get("/{user_id}")
async def get_recommendations(
    user_id: int, top_n: int = 10, recommender: IRecommender = Depends(get_recommender)
):
    return await recommender.recommend_for_user(user_id, top_n)
