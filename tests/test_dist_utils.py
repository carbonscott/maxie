import pytest
import os
import socket
import torch
import torch.distributed as dist
from unittest.mock import patch, MagicMock
from mpi4py import MPI
from datetime import timedelta

from maxie.utils.dist_utils import init_dist_env, setup_distributed

class TestDistributedUtils:
    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Clean environment variables before each test"""
        # Save any existing env vars
        saved_env = {}
        for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
            if key in os.environ:
                saved_env[key] = os.environ[key]
                del os.environ[key]

        yield

        # Restore saved env vars
        for key, value in saved_env.items():
            os.environ[key] = value

    @pytest.fixture
    def mock_mpi(self):
        """Fixture to mock MPI environment"""
        with patch('mpi4py.MPI.COMM_WORLD') as mock_comm:
            mock_comm.Get_rank.return_value = 1
            mock_comm.Get_size.return_value = 2
            mock_comm.bcast.return_value = "dummy_master"
            yield mock_comm

    @pytest.fixture
    def mock_torch_cuda(self):
        """Fixture to mock torch.cuda"""
        with patch('torch.cuda') as mock_cuda:
            mock_cuda.device_count.return_value = 4
            yield mock_cuda

    @pytest.fixture
    def mock_dist(self):
        """Fixture to mock torch.distributed"""
        with patch('torch.distributed.init_process_group') as mock_dist:
            yield mock_dist

    @pytest.fixture
    def mock_socket(self):
        """Fixture to mock socket operations"""
        with patch('socket.gethostname') as mock_hostname, \
             patch('socket.gethostbyname') as mock_hostbyname:
            mock_hostname.return_value = "test_host"
            mock_hostbyname.return_value = "123.45.67.89"
            yield (mock_hostname, mock_hostbyname)

    @pytest.fixture
    def dist_config(self):
        """Fixture for distributed configuration"""
        return {
            "dist": {
                "backend": "nccl",
                "uses_unique_world_seed": True,
                "dtype": "float16"
            }
        }

    def test_init_dist_env_multi_rank(self, mock_mpi, mock_torch_cuda, mock_socket):
        """Test initialization of distributed environment with multiple ranks"""
        init_dist_env()

        assert os.environ["WORLD_SIZE"] == "2"
        assert os.environ["RANK"] == "1"
        assert os.environ["LOCAL_RANK"] == "1"
        assert os.environ["MASTER_PORT"] == "29500"
        assert "MASTER_ADDR" in os.environ

    def test_init_dist_env_single_rank(self, mock_mpi, mock_torch_cuda, mock_socket):
        """Test initialization with single rank"""
        # Configure mock for single rank
        mock_mpi.Get_size.return_value = 1
        mock_mpi.Get_rank.return_value = 0
        mock_mpi.bcast.side_effect = Exception("Should not be called in single rank mode")

        init_dist_env()

        assert os.environ.get("WORLD_SIZE", "1") == "1"
        assert os.environ.get("MASTER_ADDR") == "127.0.0.1"

    def test_setup_distributed_multi_rank(self, mock_mpi, mock_torch_cuda, mock_dist, dist_config):
        """Test distributed setup with multiple ranks"""
        dist_env = setup_distributed(dist_config)

        assert dist_env["uses_dist"] is True
        assert dist_env["rank"] == 1
        assert dist_env["local_rank"] == 1
        assert dist_env["world_size"] == 2
        assert dist_env["backend"] == "nccl"
        assert dist_env["dtype"] == "float16"

        mock_dist.assert_called_once_with(
            backend="nccl",
            rank=1,
            world_size=2,
            timeout=timedelta(seconds=1800),
            init_method="env://"
        )

    def test_setup_distributed_single_rank(self, mock_mpi, mock_torch_cuda, mock_dist, dist_config):
        """Test distributed setup with single rank"""
        mock_mpi.Get_size.return_value = 1
        mock_mpi.Get_rank.return_value = 0

        dist_env = setup_distributed(dist_config)

        assert dist_env["uses_dist"] is False
        assert dist_env["rank"] == 0
        assert dist_env["local_rank"] == 0
        assert dist_env["world_size"] == 1
        assert dist_env["backend"] == "nccl"
        assert dist_env["dtype"] == "float16"

        mock_dist.assert_not_called()

    def test_master_addr_from_env(self, mock_mpi, mock_torch_cuda, mock_socket):
        """Test master address handling when set in environment"""
        os.environ["MASTER_ADDR"] = "192.168.1.1"

        init_dist_env()

        assert os.environ["MASTER_ADDR"] == "192.168.1.1"

    @pytest.mark.parametrize("gpu_count,rank,expected_local_rank", [
        (4, 5, 1),
        (8, 10, 2),
        (2, 3, 1),
    ])
    def test_local_rank_calculation(self, mock_mpi, mock_torch_cuda, mock_socket,
                                  gpu_count, rank, expected_local_rank):
        """Test local rank calculation with different GPU counts and ranks"""
        mock_torch_cuda.device_count.return_value = gpu_count
        mock_mpi.Get_rank.return_value = rank
        mock_mpi.Get_size.return_value = rank + 1

        init_dist_env()

        assert int(os.environ["LOCAL_RANK"]) == expected_local_rank
