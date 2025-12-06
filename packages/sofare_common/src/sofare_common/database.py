import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

POSTGRES_USER = os.getenv("POSTGRES_USER", "sofare_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "sofare_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sofare_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "timescaledb")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
DATABASE_URL_SYNC = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

engine_sync = create_engine(DATABASE_URL_SYNC, echo=False)
SessionLocal = sessionmaker(bind=engine_sync)

class Base(DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

def get_sync_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Here we would ideally run create_hypertable if it doesn't exist
        # But for now, standard tables are a good start. 
        # To make them hypertables, we need to execute raw SQL.
        # await conn.execute(text("SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);"))
