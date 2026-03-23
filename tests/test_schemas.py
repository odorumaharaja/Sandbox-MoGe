"""Unit tests for backend/schemas.py"""
import pytest
from backend.schemas import (
    TimingInfo,
    InferenceFiles,
    ImageSize,
    InferenceResponse,
    HealthCheckResponse,
    MeasurePoint,
    MeasureRequest,
    MeasureResponse,
)


class TestTimingInfo:
    """Tests for TimingInfo schema"""

    def test_timing_info_valid(self):
        """Test creating TimingInfo with valid data"""
        timing = TimingInfo(
            preprocess_s=0.5,
            inference_s=1.2,
            visualization_s=0.3,
            export_s=0.8,
            total_s=2.8
        )
        assert timing.preprocess_s == 0.5
        assert timing.inference_s == 1.2
        assert timing.visualization_s == 0.3
        assert timing.export_s == 0.8
        assert timing.total_s == 2.8


class TestInferenceFiles:
    """Tests for InferenceFiles schema"""

    def test_inference_files_all_fields(self):
        """Test creating InferenceFiles with all fields provided"""
        files = InferenceFiles(
            mesh_glb="/assets/task1/mesh.glb",
            pointcloud_ply="/assets/task1/pointcloud.ply",
            pointcloud_glb="/assets/task1/pointcloud.glb",
            depth_exr="/assets/task1/depth.exr",
            points_exr="/assets/task1/points.exr",
            mask_png="/assets/task1/mask.png",
            normal_exr="/assets/task1/normal.exr",
            image_png="/assets/task1/image.png"
        )
        assert files.mesh_glb == "/assets/task1/mesh.glb"
        assert files.depth_exr == "/assets/task1/depth.exr"
        assert files.normal_exr == "/assets/task1/normal.exr"

    def test_inference_files_optional_fields_none(self):
        """Test creating InferenceFiles with optional fields as None"""
        files = InferenceFiles(
            mesh_glb="/assets/task1/mesh.glb",
            pointcloud_ply="/assets/task1/pointcloud.ply",
            pointcloud_glb="/assets/task1/pointcloud.glb",
            depth_exr=None,
            points_exr="/assets/task1/points.exr",
            mask_png="/assets/task1/mask.png",
            normal_exr=None,
            image_png="/assets/task1/image.png"
        )
        assert files.depth_exr is None
        assert files.normal_exr is None


class TestImageSize:
    """Tests for ImageSize schema"""

    def test_image_size_valid(self):
        """Test creating ImageSize with valid data"""
        size = ImageSize(width=1920, height=1080)
        assert size.width == 1920
        assert size.height == 1080


class TestInferenceResponse:
    """Tests for InferenceResponse schema"""

    def test_inference_response_valid(self):
        """Test creating InferenceResponse with valid data"""
        timing = TimingInfo(
            preprocess_s=0.5,
            inference_s=1.2,
            visualization_s=0.3,
            export_s=0.8,
            total_s=2.8
        )
        files = InferenceFiles(
            mesh_glb="/assets/task1/mesh.glb",
            pointcloud_ply="/assets/task1/pointcloud.ply",
            pointcloud_glb="/assets/task1/pointcloud.glb",
            depth_exr="/assets/task1/depth.exr",
            points_exr="/assets/task1/points.exr",
            mask_png="/assets/task1/mask.png",
            normal_exr=None,
            image_png="/assets/task1/image.png"
        )
        image_size = ImageSize(width=800, height=600)
        
        response = InferenceResponse(
            task_id="test-task-id",
            fov=[60.0, 45.0],
            timings=timing,
            files=files,
            image_size=image_size
        )
        assert response.task_id == "test-task-id"
        assert response.fov == [60.0, 45.0]
        assert response.timings.total_s == 2.8


class TestHealthCheckResponse:
    """Tests for HealthCheckResponse schema"""

    def test_health_check_response_loaded(self):
        """Test HealthCheckResponse when model is loaded"""
        response = HealthCheckResponse(status="healthy", model_loaded=True)
        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_health_check_response_not_loaded(self):
        """Test HealthCheckResponse when model is not loaded"""
        response = HealthCheckResponse(status="unhealthy", model_loaded=False)
        assert response.status == "unhealthy"
        assert response.model_loaded is False


class TestMeasurePoint:
    """Tests for MeasurePoint schema"""

    def test_measure_point_valid(self):
        """Test creating MeasurePoint with valid coordinates"""
        point = MeasurePoint(x=100, y=200)
        assert point.x == 100
        assert point.y == 200

    def test_measure_point_zero(self):
        """Test creating MeasurePoint at origin"""
        point = MeasurePoint(x=0, y=0)
        assert point.x == 0
        assert point.y == 0


class TestMeasureRequest:
    """Tests for MeasureRequest schema"""

    def test_measure_request_valid(self):
        """Test creating MeasureRequest with valid data"""
        request = MeasureRequest(
            task_id="task-123",
            p1=MeasurePoint(x=50, y=50),
            p2=MeasurePoint(x=100, y=100)
        )
        assert request.task_id == "task-123"
        assert request.p1.x == 50
        assert request.p2.x == 100


class TestMeasureResponse:
    """Tests for MeasureResponse schema"""

    def test_measure_response_valid(self):
        """Test creating MeasureResponse with valid distance"""
        response = MeasureResponse(distance=5.5)
        assert response.distance == 5.5
        assert response.unit == "m"

    def test_measure_response_zero_distance(self):
        """Test MeasureResponse with zero distance"""
        response = MeasureResponse(distance=0.0)
        assert response.distance == 0.0
        assert response.unit == "m"

    def test_measure_response_large_distance(self):
        """Test MeasureResponse with large distance"""
        response = MeasureResponse(distance=1234.56)
        assert response.distance == 1234.56
        assert response.unit == "m"