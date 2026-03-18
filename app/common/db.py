import os
from urllib.parse import quote_plus
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

load_dotenv(override=True)


def _build_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        # 팀에서 JDBC 형식으로 전달받은 경우 SQLAlchemy 형식으로 변환
        if url.startswith("jdbc:postgresql://"):
            stripped = url.replace("jdbc:postgresql://", "", 1)

            # jdbc url에 인증정보가 없으면 DB_USER/DB_PASSWORD로 채워 넣음
            if "@" not in stripped:
                user = os.getenv("DB_USER", "").strip()
                password = os.getenv("DB_PASSWORD", "").strip()
                if not user:
                    raise ValueError("DB_USER is required when DATABASE_URL uses jdbc format.")
                encoded_password = quote_plus(password)
                return f"postgresql+asyncpg://{user}:{encoded_password}@{stripped}"

            parsed = urlparse(f"postgresql://{stripped}")
            if parsed.username and parsed.password:
                return f"postgresql+asyncpg://{stripped}"

            user = parsed.username or os.getenv("DB_USER", "").strip()
            password = parsed.password or os.getenv("DB_PASSWORD", "").strip()
            if not user:
                raise ValueError("DB_USER is required for PostgreSQL connection.")

            host = parsed.hostname or os.getenv("DB_HOST", "").strip()
            port = parsed.port or os.getenv("DB_PORT", "5432").strip()
            db_name = parsed.path.lstrip("/") or os.getenv("DB_NAME", "").strip()
            encoded_password = quote_plus(password)
            return f"postgresql+asyncpg://{user}:{encoded_password}@{host}:{port}/{db_name}"
        return url

    host = os.getenv("DB_HOST", "").strip()
    port = os.getenv("DB_PORT", "5432").strip()
    name = os.getenv("DB_NAME", "").strip()
    user = os.getenv("DB_USER", "").strip()
    password = os.getenv("DB_PASSWORD", "").strip()

    if not all([host, port, name, user]):
        raise ValueError(
            "PostgreSQL env is missing. Set DATABASE_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD."
        )

    encoded_password = quote_plus(password)
    return f"postgresql+asyncpg://{user}:{encoded_password}@{host}:{port}/{name}"


DATABASE_URL = _build_database_url()

db_ssl = os.getenv("DB_SSL", "").strip().lower()
connect_args = {"ssl": True} if db_ssl in {"1", "true", "yes", "require"} else {}

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)

SessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db():
    async with SessionLocal() as session:
        yield session
