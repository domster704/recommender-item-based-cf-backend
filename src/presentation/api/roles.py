from typing import Awaitable, Callable

from fastapi import Request, HTTPException, status

from src.shared.types.roles import Role


def require_role(*allowed_roles: Role) -> Callable[[Request], Awaitable[Role]]:
    async def _checker(request: Request) -> Role:
        role: Role | None = getattr(request.state, "role", None)
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied for role '{role}'. Required: {[r.value for r in allowed_roles]}",
            )
        return role

    return _checker
