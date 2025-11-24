from math import sqrt


class CosineSimilarity:
    @staticmethod
    def calculate(vec1: dict[int, int], vec2: dict[int, int]) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами рейтингов.

        Косинусное сходство измеряет степень похожести двух фильмов на основе
        оценок пользователей. Каждый вектор представляет собой словарь вида
        {user_id: rating}, где ключ - идентификатор пользователя, а значение -
        поставленная им оценка. Сходство вычисляется только по тем пользователям,
        которые оценили оба фильма

        Args:
            vec1: Вектор первого фильма в формате {user_id: rating}
            vec2: Вектор второго фильма в формате {user_id: rating}

        Returns:
            Значение косинусного сходства от 0.0 до 1.0.
            Возвращает 0.0, если у фильмов нет общих пользователей
            или если хотя бы один из векторов имеет нулевую норму

        References:
            - https://en.wikipedia.org/wiki/Item-item_collaborative_filtering
        """
        user_ids = set(vec1.keys()) & set(vec2.keys())
        if not user_ids:
            return 0.0

        dot = sum(vec1[user_id] * vec2[user_id] for user_id in user_ids)
        norm1 = sqrt(sum(v**2 for v in vec1.values()))
        norm2 = sqrt(sum(v**2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)
