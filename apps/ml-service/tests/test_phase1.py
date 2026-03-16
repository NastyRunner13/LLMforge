"""Tests for CRUD operations, cleaning pipeline, and API endpoints."""

import json
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════
# CRUD Tests (mocked DB)
# ═══════════════════════════════════════════════

class FakeDB:
    """Minimal mock for SQLAlchemy session."""
    def __init__(self):
        self.added = []
        self.committed = False
        self.refreshed = []
        self.deleted = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        self.refreshed.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)

    def query(self, model):
        return FakeQuery(model)


class FakeQuery:
    def __init__(self, model):
        self.model = model
        self._filters = []
        self._result = []

    def filter(self, *args):
        return self

    def order_by(self, *args):
        return self

    def offset(self, n):
        return self

    def limit(self, n):
        return self

    def first(self):
        return self._result[0] if self._result else None

    def all(self):
        return self._result


class TestCRUDCreateDataset:
    def test_create_dataset_adds_record(self):
        from app.services.crud import create_dataset
        db = FakeDB()
        ds = create_dataset(
            db,
            project_id="proj-1",
            name="test-dataset",
            original_filename="data.csv",
            file_format="csv",
            file_size_bytes=1024,
        )
        assert db.committed
        assert len(db.added) == 1
        assert db.added[0].project_id == "proj-1"
        assert db.added[0].name == "test-dataset"
        assert db.added[0].file_format == "csv"


class TestCRUDCreateTrainingRun:
    def test_create_training_run_sets_status(self):
        from app.services.crud import create_training_run
        from app.models.db_models import RunStatus
        db = FakeDB()
        run = create_training_run(
            db,
            project_id="proj-1",
            dataset_id="ds-1",
            model_config={"base_model": "test"},
            training_config={"lr": 1e-4},
        )
        assert db.committed
        assert db.added[0].status == RunStatus.QUEUED


# ═══════════════════════════════════════════════
# Cleaning Pipeline Tests
# ═══════════════════════════════════════════════

class TestCleaningPipeline:
    def test_dedup_removes_duplicates(self):
        from app.services.cleaning import dedup_node
        records = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "hello"},
        ]
        result = dedup_node(records)
        assert len(result) == 2

    def test_dedup_by_key(self):
        from app.services.cleaning import dedup_node
        records = [
            {"id": "1", "text": "hello"},
            {"id": "2", "text": "world"},
            {"id": "1", "text": "hello again"},
        ]
        result = dedup_node(records, key="id")
        assert len(result) == 2

    def test_length_filter_min_max(self):
        from app.services.cleaning import length_filter_node
        records = [
            {"text": "hi"},
            {"text": "this is a medium length text"},
            {"text": "x" * 200},
        ]
        result = length_filter_node(records, field="text", min_length=5, max_length=100)
        assert len(result) == 1
        assert result[0]["text"] == "this is a medium length text"

    def test_regex_filter_include(self):
        from app.services.cleaning import regex_filter_node
        records = [
            {"text": "python is great"},
            {"text": "java is also good"},
            {"text": "python rocks"},
        ]
        result = regex_filter_node(records, field="text", pattern="python", mode="include")
        assert len(result) == 2

    def test_regex_filter_exclude(self):
        from app.services.cleaning import regex_filter_node
        records = [
            {"text": "keep this"},
            {"text": "remove bad-word here"},
            {"text": "also keep"},
        ]
        result = regex_filter_node(records, field="text", pattern="bad-word", mode="exclude")
        assert len(result) == 2

    def test_pii_redact_email(self):
        from app.services.cleaning import pii_redact_node
        records = [{"text": "Contact me at john@example.com for info"}]
        result = pii_redact_node(records, fields=["text"])
        assert "john@example.com" not in result[0]["text"]
        assert "[REDACTED]" in result[0]["text"]

    def test_pii_redact_phone(self):
        from app.services.cleaning import pii_redact_node
        records = [{"text": "Call me at 555-123-4567"}]
        result = pii_redact_node(records, fields=["text"])
        assert "555-123-4567" not in result[0]["text"]

    def test_pipeline_runs_sequentially(self):
        from app.services.cleaning import run_pipeline
        records = [
            {"text": "hello"},
            {"text": "hello"},
            {"text": "hi"},
            {"text": "x"},
        ]
        nodes = [
            {"node_type": "dedup", "params": {}},
            {"node_type": "length_filter", "params": {"min_length": 2}},
        ]
        result = run_pipeline(records, nodes)
        assert len(result) == 2  # dedup removes 1, length filter removes "x"


# ═══════════════════════════════════════════════
# Rate Limiter Tests
# ═══════════════════════════════════════════════

class TestRateLimiter:
    def test_limiter_allows_under_limit(self):
        from app.core.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)
        # With no Redis, should pass through (graceful degradation)
        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        result = limiter.check(mock_request)
        assert result is True
