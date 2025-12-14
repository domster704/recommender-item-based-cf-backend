from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from src.domain.entities.movie_lens.raitings import Rating
from src.infrastructure.config.settings import settings, SVDConfig
from src.infrastructure.services.recommender_module.recommender.funk_svd.funk_svd_recommender import (
    FunkSVDTorchRecommender,
)


class FunkSVDModel(nn.Module):
    """
    Модель представления пользователей и фильмов как латентных векторов.

    Модель пытается аппроксимировать следующее:
        r_ui = mu + b_u + b_i + p_u * q_i,
    где
    * mu - это глобальное среднее рейтинга
    * b_u - смещение пользователя
    * b_i - смещение фильма
    * p_u, q_i - латентные векторы пользователя и фильма
    * p_u * q_i - степень того, насколько предпочтения пользователя совпадают с особенностями фильма
    * r_ui - прогноз рейтинга пользователя u для фильма i
    """

    def __init__(
        self, num_users: int, num_items: int, factors: int, global_mean: float = 0.0
    ):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, factors)
        self.item_factors = nn.Embedding(num_items, factors)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_factors.weight, std=settings.svd.init_std)
        nn.init.normal_(self.item_factors.weight, std=settings.svd.init_std)

        self.global_mean = global_mean

    def forward(self, user_ids, item_ids):
        p_u = self.user_factors(user_ids)
        q_i = self.item_factors(item_ids)

        dot = torch.sum(p_u * q_i, dim=1)

        b_u = self.user_bias(user_ids).squeeze()
        b_i = self.item_bias(item_ids).squeeze()

        return self.global_mean + b_u + b_i + dot


class TorchMFTrainer:
    def __init__(
        self, factors: int = 30, epochs: int = 10, lr: float = 0.001, device=None
    ):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_model(self) -> FunkSVDTorchRecommender:
        """Загружает модель из файла."""
        data = torch.load(settings.funk_svd_model_path, map_location=self.device)

        model = FunkSVDModel(
            num_users=len(data["user_to_idx"]),
            num_items=len(data["item_to_idx"]),
            factors=self.factors,
            global_mean=data["global_mean"],
        ).to(self.device)

        model.load_state_dict(data["model_state"])

        return FunkSVDTorchRecommender(
            model=model,
            user_items=data["user_items"],
            user_to_idx=data["user_to_idx"],
            item_to_idx=data["item_to_idx"],
            popular_items=data["popular_items"],
            trainer=self,
        )

    def save_model(
        self,
        model: FunkSVDModel,
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
        user_items: dict[int, set[int]],
        popular_items: list[int],
    ) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
                "global_mean": model.global_mean,
                "user_items": user_items,
                "popular_items": popular_items,
            },
            settings.funk_svd_model_path,
        )

    @staticmethod
    def build_indices(
        ratings: list[Rating],
    ) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
        users = sorted({r.user.id for r in ratings})
        items = sorted({r.movie.id for r in ratings})

        user_to_idx = {u: i for i, u in enumerate(users)}
        item_to_idx = {m: j for j, m in enumerate(items)}

        return users, items, user_to_idx, item_to_idx

    def build_tensors(
        self,
        ratings: list[Rating],
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_ids = torch.tensor(
            [user_to_idx[r.user.id] for r in ratings], device=self.device
        )
        item_ids = torch.tensor(
            [item_to_idx[r.movie.id] for r in ratings], device=self.device
        )
        values = torch.tensor([float(r.rating) for r in ratings], device=self.device)

        return user_ids, item_ids, values

    def train(
        self,
        model: FunkSVDModel,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        dataset = torch.utils.data.TensorDataset(user_ids, item_ids, values)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            for u, i, r in loader:
                pred = model(u, i)
                loss = loss_fn(pred, r)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{self.epochs}, loss={loss.item():.4f}")

    @staticmethod
    def build_user_items(ratings: list[Rating]) -> dict[int, set[int]]:
        users = {}
        for r in ratings:
            users.setdefault(r.user.id, set()).add(r.movie.id)
        return users

    @staticmethod
    def build_popular_items(ratings: list[Rating]) -> list[int]:
        return [
            movie for movie, _ in Counter(r.movie.id for r in ratings).most_common()
        ]

    def fit(self, ratings: list[Rating]) -> FunkSVDTorchRecommender:
        # Если модель уже есть — загружаем и выходим
        if settings.funk_svd_model_path.exists():
            return self.load_model()

        users, items, user_to_idx, item_to_idx = self.build_indices(ratings)

        user_ids, item_ids, values = self.build_tensors(
            ratings, user_to_idx, item_to_idx
        )

        model = FunkSVDModel(
            num_users=len(users),
            num_items=len(items),
            factors=self.factors,
        ).to(self.device)
        model.global_mean = values.mean().item()

        self.train(model, user_ids, item_ids, values)

        user_items = self.build_user_items(ratings)
        popular_items = self.build_popular_items(ratings)

        self.save_model(model, user_to_idx, item_to_idx, user_items, popular_items)

        return FunkSVDTorchRecommender(
            model=model,
            user_items=user_items,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            popular_items=popular_items,
            trainer=self,
        )

    def online_update(
        self,
        model: nn.Module,
        rating: Rating,
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
    ) -> None:
        """Онлайн-обучение: один шаг FunkSVD для одного пользователя."""
        cfg: SVDConfig = settings.svd

        user_id = rating.user.id
        movie_id = rating.movie.id
        value = float(rating.rating)

        # создаём нового пользователя, если его нет
        if user_id not in user_to_idx:
            self._add_new_user(model, user_id, user_to_idx)

        u_idx = user_to_idx[user_id]
        i_idx = item_to_idx[movie_id]

        device = next(model.parameters()).device

        u = torch.tensor([u_idx], device=device)
        i = torch.tensor([i_idx], device=device)
        r_true = torch.tensor([value], device=device)

        model.train()

        optimizer = optim.SGD(
            [
                model.user_factors.weight,
                model.user_bias.weight,
            ],
            lr=cfg.online_lr,
        )

        loss_fn = nn.MSELoss()

        for _ in range(cfg.online_steps):
            optimizer.zero_grad()

            pred = model(u, i)

            p_u = model.user_factors(u)
            q_i = model.item_factors(i)

            loss = loss_fn(pred, r_true)
            loss += cfg.online_reg * (p_u.norm() + q_i.norm())

            loss.backward()
            optimizer.step()

        model.eval()

    def _add_new_user(self, model, user_id: int, user_to_idx: dict[int, int]) -> None:
        new_idx = len(user_to_idx)
        user_to_idx[user_id] = new_idx

        device = next(model.parameters()).device
        cfg: SVDConfig = settings.svd

        with torch.no_grad():
            new_factors = torch.normal(
                mean=0,
                std=cfg.init_std,
                size=(1, model.user_factors.embedding_dim),
                device=device,
            )

            new_bias = torch.zeros((1, 1), device=device)

            model.user_factors.weight = nn.Parameter(
                torch.cat([model.user_factors.weight, new_factors], dim=0)
            )
            model.user_bias.weight = nn.Parameter(
                torch.cat([model.user_bias.weight, new_bias], dim=0)
            )
