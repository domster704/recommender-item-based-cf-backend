import csv
from datetime import datetime
from pathlib import Path

from src.domain.entities.movie_lens.genre import Genre
from src.domain.entities.movie_lens.movie import Movie
from src.domain.entities.movie_lens.occupation import Occupation
from src.domain.entities.movie_lens.raitings import Rating
from src.domain.entities.movie_lens.user import User, UserGender
from src.infrastructure.db.models import OccupationORM
from src.infrastructure.db.models.movie_lens.links import MovieGenreLink
from src.infrastructure.db.uow import UnitOfWork
from src.infrastructure.repositories.genre import GenreRepository
from src.infrastructure.repositories.movie import MovieRepository
from src.infrastructure.repositories.occupation import OccupationRepository
from src.infrastructure.repositories.rating import RatingRepository
from src.infrastructure.repositories.user import UserRepository


class MovieLensImporter:
    def __init__(self, base_path: Path):
        self.genre_file = base_path / "u.genre"
        self.occupation_file = base_path / "u.occupation"
        self.user_file = base_path / "u.user"
        self.movie_file = base_path / "u.item"
        self.rating_file = base_path / "u.data"
        self.imdb_basics = base_path / "title.basics.tsv"

    async def import_all(self):
        async with UnitOfWork() as uow:
            await self._import_genres(uow)
            await self._import_occupations(uow)
            await self._import_users(uow)
            await self._import_movies(uow)
            await self._import_ratings(uow)

    async def _import_genres(self, uow: UnitOfWork):
        repo = GenreRepository(uow)
        if await repo.get_all():
            return

        with open(self.genre_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")

            for name, id_value in reader:
                genre = Genre(
                    id=int(id_value),
                    name=name,
                )
                await repo.add(genre, commit=False)

        await uow.commit()

    async def _import_occupations(self, uow: UnitOfWork):
        repo = OccupationRepository(uow)
        if await repo.get_all():
            return

        with open(self.occupation_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                name = line.strip()
                occupation = Occupation(id=idx, name=name)
                await repo.add(occupation, commit=False)

        await uow.commit()

    async def _import_users(self, uow: UnitOfWork):
        user_repo = UserRepository(uow)
        occ_repo = OccupationRepository(uow)
        if await user_repo.get_all():
            return

        with open(self.user_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                user_id, age, gender_raw, occupation_name, zipcode = row

                occ: Occupation | None = await occ_repo.get(
                    reference=occupation_name,
                    field_search=OccupationORM.name.name,
                )

                user = User(
                    id=int(user_id),
                    age=int(age),
                    gender=UserGender.M if gender_raw == "M" else UserGender.F,
                    occupation=occ,
                )

                await user_repo.add(user, commit=False)

        await uow.commit()

    async def _import_movies(self, uow: UnitOfWork):
        movie_repo = MovieRepository(uow)
        if await movie_repo.get_all():
            return

        genres = await GenreRepository(uow).get_all()

        with open(self.movie_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")

            for row in reader:
                movie_id = int(row[0])

                title = row[1]

                release_date_raw = row[2]
                video_date_raw = row[3]
                imdb_url = row[4]
                genre_flags = row[5:]

                release_date = (
                    datetime.strptime(release_date_raw, "%d-%b-%Y").date()
                    if release_date_raw
                    else None
                )
                video_release_date = (
                    datetime.strptime(video_date_raw, "%d-%b-%Y").date()
                    if video_date_raw
                    else None
                )

                genre_ids = [
                    genre.id
                    for flag, genre in zip(genre_flags, genres)
                    if int(flag) == 1
                ]

                movie = Movie(
                    id=movie_id,
                    title=title,
                    release_date=release_date,
                    video_release_date=video_release_date,
                    imdb_url=imdb_url,
                    genres=[genre for genre in genres if genre.id in genre_ids],
                )

                await movie_repo.add(movie, commit=False)

                for gid in genre_ids:
                    uow.session.add(MovieGenreLink(movie_id=movie_id, genre_id=gid))

        await uow.commit()

    async def _import_ratings(self, uow: UnitOfWork):
        rating_repo = RatingRepository(uow)
        if await rating_repo.get_all():
            return

        user_repo = UserRepository(uow)
        movie_repo = MovieRepository(uow)

        users = await user_repo.get_all()
        users_map = {u.id: u for u in users}

        movies = await movie_repo.get_all()
        movies_map = {m.id: m for m in movies}

        with open(self.rating_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")

            for user_id, movie_id, rating, timestamp in reader:
                rating_obj = Rating(
                    user=users_map[int(user_id)],
                    movie=movies_map[int(movie_id)],
                    rating=int(rating),
                    timestamp=int(timestamp),
                )

                await rating_repo.add(rating_obj, commit=False)

        await uow.commit()
