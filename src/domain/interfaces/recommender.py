from typing import Protocol, Callable, Awaitable

from src.domain.entities.movie_lens.movie import Movie
from src.domain.entities.movie_lens.raitings import Rating


class IRecommender(Protocol):
    async def recommend_for_user(self, user_id: int, top_n: int = 10) -> list[int]:
        """
        Формирует рекомендации для пользователя

        Args:
            user_id: Идентификатор пользователя
            top_n: Количество фильмов, которые нужно вернуть

        Returns:
            Список идентификаторов, рекомендованных пользователю
        """
        ...

    async def update_for_rating(self, rating: Rating) -> None:
        """
        Обновляет рекомендационную модель после появления нового рейтинга.

        Обновляется только малая часть матрицы сходств,
        а не пересчитывается весь датасет. Это позволяет эффективно
        поддерживать актуальность модели в режиме online.

        Args:
            rating (Rating): Новый или обновлённый рейтинг

        Returns:
            None
        """
        ...


class IRecommenderBuilder(Protocol):
    async def build(
        self,
        ratings_loader: Callable[[], Awaitable[list[Rating]]],
        movies_loader: Callable[[], Awaitable[list[Movie]]],
    ) -> IRecommender:
        """
        Формирует внутреннюю модель рекомендаций на основе переданных данных.

        Метод загружает данные через переданные загрузчики, заполняет структуру
        пользовательских рейтингов и формирует матрицу сходства фильмов.
        Если доступен кеш, данные читаются из него и сохраняются после расчёта

        Args:
            ratings_loader: Асинхронная функция, возвращающая список всех рейтингов
            movies_loader: Асинхронная функция, возвращающая список всех фильмов

        Returns:
            None
        """
        ...

    async def build_funk_svd(
        self,
        ratings_loader: Callable[[], Awaitable[list[Rating]]],
        factors: int = 30,
        epochs: int = 15,
        learning_rate: float = 0.01,
    ) -> IRecommender:
        """Новый вариант – рекомендатель на основе матричного разложения FunkSVD."""
        ...
