"""Integration tests for Dataset and Training API endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from app.models.db_models import DatasetStatus, RunStatus


class TestDatasetAPI:
    """Tests for /api/datasets/* endpoints."""

    def test_upload_initiate(self, client, mock_s3):
        """POST /api/datasets/upload should return presigned URL and dataset ID."""
        response = client.post(
            "/api/datasets/upload",
            json={
                "project_id": "test-project",
                "filename": "test_data.csv",
                "content_type": "text/csv",
                "file_size_bytes": 1024,
            },
            headers={"X-Internal-Key": "test-secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "upload_url" in data
        assert "dataset_id" in data

    def test_get_dataset_not_found(self, client):
        """GET /api/datasets/{id} should 404 for non-existent dataset."""
        response = client.get(
            "/api/datasets/nonexistent-id",
            headers={"X-Internal-Key": "test-secret"},
        )
        assert response.status_code == 404

    def test_list_datasets_empty(self, client):
        """GET /api/datasets?project_id=xxx should return empty list."""
        response = client.get(
            "/api/datasets?project_id=test-project",
            headers={"X-Internal-Key": "test-secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data.get("datasets", []), list)

    def test_delete_dataset_not_found(self, client):
        """DELETE /api/datasets/{id} should return 404 for missing dataset."""
        response = client.delete(
            "/api/datasets/nonexistent-id",
            headers={"X-Internal-Key": "test-secret"},
        )
        assert response.status_code == 404


class TestTrainingAPI:
    """Tests for /api/training/runs/* endpoints."""

    def test_create_training_run(self, client, mock_celery):
        """POST /api/training/runs should create a new training run."""
        # First create a dataset for the run
        upload_res = client.post(
            "/api/datasets/upload",
            json={
                "project_id": "test-project",
                "filename": "data.jsonl",
                "content_type": "application/jsonl",
                "file_size_bytes": 2048,
            },
            headers={"X-Internal-Key": "test-secret"},
        )
        dataset_id = upload_res.json().get("dataset_id", "test-ds")

        response = client.post(
            "/api/training/runs",
            json={
                "project_id": "test-project",
                "dataset_id": dataset_id,
                "experiment_name": "Test Run",
                "model_config": {
                    "base_model": "meta-llama/Llama-3-8B",
                    "training_method": "lora",
                },
                "training_config": {
                    "num_epochs": 3,
                    "learning_rate": 2e-4,
                    "per_device_train_batch_size": 4,
                    "lora_r": 16,
                },
            },
            headers={"X-Internal-Key": "test-secret"},
        )
        assert response.status_code in (200, 201)
        data = response.json()
        assert "run_id" in data or "id" in data

    def test_list_training_runs_empty(self, client):
        """GET /api/training/runs should return list."""
        response = client.get(
            "/api/training/runs?project_id=test-project",
            headers={"X-Internal-Key": "test-secret"},
        )
        assert response.status_code == 200


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """GET /health should return healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "ok")
