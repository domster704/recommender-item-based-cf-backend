from collections import defaultdict

from src.domain.entities.movie_lens.raitings import Rating
from src.domain.interfaces.recommender import IRecommender
from src.infrastructure.services.recommender_module.similarity.cosine import (
    CosineSimilarity,
)
from src.infrastructure.services.recommender_module.storage.ratings_storage import (
    RatingsStorage,
)


class ItemBasedCFRecommender(IRecommender):
    """
    References:
       - https://ru.wikipedia.org/wiki/Коллаборативная_фильтрация
       - https://en.wikipedia.org/wiki/Item-item_collaborative_filtering
    """

    def __init__(
        self,
        similarity_matrix: dict[int, dict[int, float]],
        ratings_storage: RatingsStorage,
    ) -> None:
        """
        Args:
            similarity_matrix: Матрица сходства фильмов вида
                {movie_id: {other_movie_id: similarity}}
            ratings_storage: Хранилище пользовательских рейтингов
        """
        self.similarity: dict[int, dict[int, float]] = similarity_matrix
        self.storage: RatingsStorage = ratings_storage

    async def recommend_for_user(self, user_id: int, top_n: int = 10) -> list[int]:
        user_movies: dict[int, int] = self.storage.get_user_movies(user_id)
        if not user_movies:
            return self.storage.popular(top_n)

        scores: dict[int, float] = defaultdict(float)
        weights: dict[int, float] = defaultdict(float)

        for movie_id, rating in user_movies.items():
            for other_movie, similarity in self.similarity.get(movie_id, {}).items():
                if other_movie in user_movies:
                    continue

                # чем выше рейтинг фильма Х и чем сильнее Х похож на Y => тем больше вклад Х в оценку Y
                scores[other_movie] += similarity * rating
                weights[other_movie] += abs(similarity)

        ranked: list[tuple[int, float]] = [
            (mid, scores[mid] / weights[mid]) for mid in scores if weights[mid] > 0
        ]

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in ranked[:top_n]]

    async def update_for_rating(self, rating: Rating) -> None:
        user_id: int = rating.user.id
        movie_id: int = rating.movie.id

        self.storage.update(rating)

        user_movies: dict[int, int] = self.storage.get_user_movies(user_id)
        for other_id in user_movies:
            if other_id == movie_id:
                continue

            vector1: dict[int, int] = self.storage.get_movie_vector(movie_id)
            vector2: dict[int, int] = self.storage.get_movie_vector(other_id)

            similarity: float = CosineSimilarity.calculate(vector1, vector2)

            if similarity > 0:
                self.similarity[movie_id][other_id] = similarity
                self.similarity[other_id][movie_id] = similarity
            else:
                self.similarity[movie_id].pop(other_id, None)
                self.similarity[other_id].pop(movie_id, None)
