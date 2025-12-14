import torch
from torch import nn, optim
from torch.cuda import device

from src.domain.entities.movie_lens.raitings import Rating
from src.domain.interfaces.recommender import IRecommender
from src.infrastructure.config.settings import settings, SVDConfig


class FunkSVDTorchRecommender(IRecommender):
    def __init__(
        self,
        model: nn.Module,
        user_items: dict[int, set[int]],
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
        popular_items: list[int],
        trainer: "TorchMFTrainer",
    ):
        self.model = model.eval()
        self.user_items = user_items
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.popular_items = popular_items
        self.device: device = next(model.parameters()).device

        self.cfg: SVDConfig = settings.svd
        self.trainer = trainer

    def _predict(self, user_id: int, movie_id: int) -> float:
        """
        Предсказание рейтинга для пары (u, i)

        """

        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[movie_id]

        user = torch.tensor([u_idx], device=self.device)
        item = torch.tensor([i_idx], device=self.device)

        with torch.no_grad():
            return float(self.model(user, item).cpu())

    async def recommend_for_user(self, user_id: int, top_n: int = 10) -> list[int]:
        # Холодный старт
        if user_id not in self.user_to_idx:
            return self.popular_items[:top_n]

        watched: set[int] = self.user_items.get(user_id, set())
        all_items: list[int] = list(self.item_to_idx.keys())

        scores: list[tuple[int, float]] = []
        for item_id in all_items:
            if item_id in watched:
                continue

            score: float = self._predict(user_id, item_id)
            scores.append((item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in scores[:top_n]]

    async def update_for_rating(self, rating: Rating) -> None:
        user_id = rating.user.id
        movie_id = rating.movie.id

        self.user_items.setdefault(user_id, set()).add(movie_id)

        self.trainer.online_update(
            model=self.model,
            rating=rating,
            user_to_idx=self.user_to_idx,
            item_to_idx=self.item_to_idx,
        )
