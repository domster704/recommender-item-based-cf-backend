from collections import defaultdict, Counter

from src.domain.entities.movie_lens.raitings import Rating


class RatingsStorage:
    def __init__(self):
        self.users: dict[int, dict[int, int]] = defaultdict(dict)

    def set_users(self, users: dict[int, dict[int, int]]):
        self.users = users

    def fill(self, ratings: list[Rating]):
        for r in ratings:
            self.users[r.user.id][r.movie.id] = r.rating

    def update(self, rating: Rating):
        self.users[rating.user.id][rating.movie.id] = rating.rating

    def get_user_movies(self, user_id: int) -> dict[int, int]:
        return self.users.get(user_id, {})

    def get_movie_vector(self, movie_id: int) -> dict[int, int]:
        vector = {}
        for u, movies in self.users.items():
            if movie_id in movies:
                vector[u] = movies[movie_id]
        return vector

    def popular(self, top_n: int) -> list[int]:
        counter = Counter()
        for movies in self.users.values():
            counter.update(movies.keys())

        return [mid for mid, _ in counter.most_common(top_n)]
