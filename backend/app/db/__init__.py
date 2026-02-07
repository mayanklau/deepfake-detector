"""Database connection and session management."""
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

engine = create_async_engine(
    f"postgresql+asyncpg://{settings.database.POSTGRES_USER}:{settings.database.POSTGRES_PASSWORD}"
    f"@{settings.database.POSTGRES_HOST}:{settings.database.POSTGRES_PORT}/{settings.database.POSTGRES_DB}",
    echo=settings.DEBUG,
    pool_size=settings.database.POOL_SIZE,
    max_overflow=settings.database.MAX_OVERFLOW,
    pool_timeout=settings.database.POOL_TIMEOUT,
    pool_recycle=settings.database.POOL_RECYCLE,
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_session() -> AsyncSession:
    """Dependency to get database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_database():
    """Initialize database - create tables."""
    from app.models.database import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully")

async def close_database():
    """Close database connections."""
    await engine.dispose()
    logger.info("Database connections closed")
