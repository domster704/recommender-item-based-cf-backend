from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from src.infrastructure.db.db import init_db
from src.presentation.api.router_v1 import api_v1_router
from src.presentation.dependencies.movie_lens.movie_lens_impot import get_movie_lens_import_use_case


# from starlette.middleware.sessions import SessionMiddleware

# from src.presentation.middlewares.auth import AuthMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    movie_lens_import_use_case = get_movie_lens_import_use_case()
    movie_lens_path = Path(
        r"C:\Users\isupov\YandexDisk\Programming\University\ERS\async-collaborative-filtering\async-collaborative-filtering-backend\assets"
    )
    await movie_lens_import_use_case.execute(movie_lens_path)

    yield

    pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.add_middleware(AuthMiddleware)
app.include_router(api_v1_router)


@app.get("/health")
async def health():
    return {"health": "OK"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
