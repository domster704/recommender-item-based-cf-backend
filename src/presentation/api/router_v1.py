from fastapi import APIRouter

# from src.presentation.api.v1.calendar_router import calendar_router
# from src.presentation.api.v1.google_router import google_router
# from src.presentation.api.v1.payment_router import payment_router
# from src.presentation.api.v1.security_router import security_router
# from src.presentation.api.v1.subscription_router import subscription_router
# from src.presentation.api.v1.user_router import user_router

api_v1_router = APIRouter(prefix="/api/v1")
# api_v1_router.include_router(calendar_router)
# api_v1_router.include_router(user_router)
# api_v1_router.include_router(security_router)
# api_v1_router.include_router(subscription_router)
# api_v1_router.include_router(google_router)
# api_v1_router.include_router(payment_router)
