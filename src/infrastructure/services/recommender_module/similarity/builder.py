from collections import defaultdict
from typing import Callable

from src.domain.entities.movie_lens.movie import Movie
from .cosine import CosineSimilarity


class SimilarityMatrixBuilder:
    """Строит матрицу сходства фильмов на основе пользовательских рейтингов.

    Класс формирует векторы фильмов вида movie_id → {user_id → rating}
    и вычисляет косинусное сходство для всех пар фильмов.
    """

    def __init__(
        self,
        similarity_function: Callable[
            [dict[int, int], dict[int, int]], float
        ] = CosineSimilarity.calculate,
    ) -> None:
        self.similarity_function = similarity_function

    def build(
        self,
        user_ratings: dict[int, dict[int, int]],
        movies: list[Movie],
    ) -> dict[int, dict[int, float]]:
        """Строит матрицу сходства фильмов.

        Создаёт структуру вида movie_id → {other_movie_id → similarity}.
        Сходство вычисляется только для тех фильмов, по которым есть рейтинги.
        Пары с нулевым сходством не сохраняются.

        Args:
            user_ratings: Словарь пользовательских оценок,
                структура user_id → {movie_id → rating}.
            movies: Список всех фильмов.

        Returns:
            Матрица сходства фильмов вида:
                {movie_id: {other_movie_id: similarity}}.
        """
        movie_vectors: dict[int, dict[int, int]] = defaultdict(dict)

        for user_id, movie_data in user_ratings.items():
            for movie_id, rating in movie_data.items():
                movie_vectors[movie_id][user_id] = rating

        matrix: dict[int, dict[int, float]] = defaultdict(dict)
        movie_ids: list[int] = [movie.id for movie in movies]

        for i, m1 in enumerate(movie_ids):
            v1 = movie_vectors.get(m1, {})

            for m2 in movie_ids[i + 1 :]:
                v2 = movie_vectors.get(m2, {})

                sim = self.similarity_function(v1, v2)

                if sim > 0:
                    matrix[m1][m2] = sim
                    matrix[m2][m1] = sim

        return matrix
