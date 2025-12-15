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
        """Загружает модель из файла"""
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
        # Если модель уже есть - загружаем и выходим
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
        model: nn.Module,  # ожидается FunkSVDModel
        rating,  # Rating
        user_to_idx: dict[int, int],
        item_to_idx: dict[int, int],
        update_item: bool = True,
    ) -> None:
        """
        Онлайн-обучение (адаптация под новый рейтинг).
        По умолчанию обновляет ТОЛЬКО пользователя (p_u и b_u).
        Опционально может обновлять ещё и фильм (q_i и b_i).
        """
        cfg = settings.svd

        user_id = rating.user.id
        movie_id = rating.movie.id
        r_true_val = float(rating.rating)

        device = next(model.parameters()).device

        if user_id not in user_to_idx:
            self._add_new_user(model, user_id, user_to_idx)

        # фильм должен существовать
        if movie_id not in item_to_idx:
            return

        u_idx: int = user_to_idx[user_id]
        i_idx: int = item_to_idx[movie_id]

        mu = torch.tensor(float(model.global_mean), device=device)

        with torch.no_grad():
            p0 = model.user_factors.weight[u_idx].detach().clone()
            bu0 = model.user_bias.weight[u_idx].detach().clone()

            q0 = model.item_factors.weight[i_idx].detach().clone()
            bi0 = model.item_bias.weight[i_idx].detach().clone()

        p_u = nn.Parameter(p0)
        b_u = nn.Parameter(bu0)

        params = [p_u, b_u]

        if update_item:
            q_i = nn.Parameter(q0)
            b_i = nn.Parameter(bi0)
            params += [q_i, b_i]
        else:
            q_i = q0.detach()
            b_i = bi0.detach()

        optimizer = optim.SGD(params, lr=cfg.online_lr)

        r_true = torch.tensor(r_true_val, device=device)

        model.eval()

        for _ in range(cfg.online_steps):
            optimizer.zero_grad(set_to_none=True)

            # pred = mu + b_u + b_i + p_u @ q_i
            dot = (p_u * q_i).sum()
            pred = mu + b_u.view(()) + b_i.view(()) + dot

            err = pred - r_true
            mse = err * err

            reg = cfg.online_reg * (p_u.pow(2).sum() + b_u.pow(2).sum())
            if update_item:
                reg = reg + cfg.online_reg * (q_i.pow(2).sum() + b_i.pow(2).sum())

            loss = mse + reg
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.user_factors.weight[u_idx].copy_(p_u.data)
            model.user_bias.weight[u_idx].copy_(b_u.data)

            if update_item:
                model.item_factors.weight[i_idx].copy_(q_i.data)
                model.item_bias.weight[i_idx].copy_(b_i.data)

    def _add_new_user(self, model, user_id: int, user_to_idx: dict[int, int]) -> None:
        """
        Добавляет нового пользователя корректно: расширяет nn.Embedding целиком.
        """
        new_idx: int = len(user_to_idx)
        user_to_idx[user_id] = new_idx

        device = next(model.parameters()).device
        cfg: SVDConfig = settings.svd

        model.user_factors = self._expand_embedding(
            model.user_factors,
            n_new=1,
            init="normal",
            init_std=cfg.init_std,
            device=device,
        )
        model.user_bias = self._expand_embedding(
            model.user_bias,
            n_new=1,
            init="zeros",
            init_std=0.0,
            device=device,
        )

    @staticmethod
    def _expand_embedding(
        emb: nn.Embedding,
        n_new: int,
        init: str,
        init_std: float,
        device: torch.device,
    ) -> nn.Embedding:
        """
        Возвращает новый Embedding размером (old + n_new),
        копирует старые веса, инициализирует новые строки
        """
        old_n, dim = emb.weight.shape
        new_emb = nn.Embedding(old_n + n_new, dim).to(device)

        with torch.no_grad():
            new_emb.weight[:old_n].copy_(emb.weight.data)

            if init == "normal":
                new_emb.weight[old_n:].normal_(mean=0.0, std=float(init_std))
            elif init == "zeros":
                new_emb.weight[old_n:].zero_()
            else:
                raise ValueError(f"Unknown init: {init}")

        return new_emb
