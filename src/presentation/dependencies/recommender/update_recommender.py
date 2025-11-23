from fastapi import Depends

from src.application.usecase.recommender.update_recommender import (
    UpdateRecommenderUseCase,
)
from src.domain.interfaces.recommender import IRecommender
from src.presentation.dependencies.recommender.get_recommender import get_recommender


def get_update_recommender_use_case(
    recommender: IRecommender = Depends(get_recommender),
) -> UpdateRecommenderUseCase:
    return UpdateRecommenderUseCase(recommender=recommender)
