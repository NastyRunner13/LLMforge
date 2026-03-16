"""Shared test fixtures for ML service tests.

Provides:
- In-memory SQLite test database session
- Mocked S3 client
- Mocked Celery app (eager mode)
- FastAPI TestClient
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set test env before importing app modules
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["S3_ACCESS_KEY"] = "test"
os.environ["S3_SECRET_KEY"] = "test"
os.environ["INTERNAL_API_SECRET"] = "test-secret"

from app.core.database import Base
from app.models.db_models import (
    Dataset, TrainingRun, Checkpoint, RunMetric, Model, Endpoint,
)


@pytest.fixture(scope="session")
def engine():
    """Create a test database engine."""
    engine = create_engine("sqlite:///./test.db", echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db(engine):
    """Create a fresh database session for each test."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture()
def mock_s3():
    """Mock S3 client to prevent real S3 operations."""
    mock = MagicMock()
    mock.generate_presigned_url.return_value = "https://mock-s3.example.com/upload"
    mock.put_object.return_value = {}
    mock.delete_object.return_value = {}
    mock.get_object.return_value = {"Body": MagicMock(read=lambda: b'{"text": "test"}')}
    with patch("app.core.s3.s3_client", mock):
        yield mock


@pytest.fixture()
def mock_celery():
    """Mock Celery to run tasks synchronously."""
    with patch("app.core.celery_app.celery_app") as mock_app:
        mock_app.send_task.return_value = MagicMock(id="test-task-id")
        yield mock_app


@pytest.fixture()
def client(db, mock_s3, mock_celery):
    """Create a FastAPI TestClient with mocked dependencies."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core.database import get_db

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
