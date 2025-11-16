from __future__ import annotations

from fastapi import (
    Request,
    Response,
    status,
)
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from src.application.dto.user.user import UserDTO
from src.application.providers.uow import uow_context
from src.application.usecase.user.get_user import GetUserUseCase
from src.infrastructure.config.settings import settings
from src.shared.types.roles import Role

SECRET_KEY = settings.jwt.secret.get_secret_value()
ALGORITHM = settings.jwt.algorithm


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware, проводящий JWT-аутентификацию для каждого запроса.

    * Проверяет наличие и корректность заголовка ``Authorization: Bearer …``.
    * Декодирует токен и сохраняет payload в ``request.state.jwt_payload``.
    * Пропускает «белый список» эндпоинтов (документация, логин и т.д.).
    """

    # Эндпоинты, которые не требуют токена
    _EXEMPT: set[str] = {
        "/api/v1/import",
        "/api/v1/auth/login",
        "/api/v1/google/auth",
        "/api/v1/google/check-auth",
        "/api/v1/google/oauth/callback",
        "/docs",
        "/api/v1/docs/oauth2-redirect",
        "/health",
        "/openapi.json",
    }

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Основной обработчик middleware.

        Args:
            request: входящий запрос.
            call_next: функция передачи управления следующему медлу.

        Returns:
            HTTP-ответ.

        Raises:
            HTTPException: 401, если токен отсутствует или некорректен.
        """
        if request.method not in ["GET", "POST", "DELETE", "PUT"]:
            return await call_next(request)

        if request.url.path in self._EXEMPT:
            return await call_next(request)

        header: str | None = request.headers.get("Authorization")
        if not header or not header.startswith("Bearer "):
            return JSONResponse(
                {"detail": "Missing bearer token"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        token: str = header.removeprefix("Bearer ").strip()
        if token in settings.security.apikeys:
            request.state.role = Role.SYSTEM
            request.state.user = None
            return await call_next(request)

        try:
            payload: dict = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            return JSONResponse(
                {"detail": "Invalid token"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        user: UserDTO | None = None
        user_id: str = payload["sub"]

        async with uow_context() as uow:
            get_user_uc = GetUserUseCase(uow)
            user: UserDTO | None = await get_user_uc.execute(user_id)

        if not user:
            return JSONResponse(
                {"detail": "User not found"}, status_code=status.HTTP_401_UNAUTHORIZED
            )

        request.state.user = user
        request.state.role = getattr(user, "role", Role.UNSUBSCRIBED)
        return await call_next(request)
