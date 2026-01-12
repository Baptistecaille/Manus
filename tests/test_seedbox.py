"""
Tests for SeedboxExecutor.

These tests require Docker to be running with the manus-sandbox container.
Run: docker-compose up -d
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seedbox_executor import SeedboxExecutor, DockerExecutor


class TestDockerExecutor:
    """Tests for Docker-based execution."""

    @pytest.fixture
    def executor(self):
        """Create a Docker executor for testing."""
        return DockerExecutor(container_name="manus-sandbox")

    def test_execute_echo(self, executor):
        """Test basic echo command."""
        result = executor.execute_bash("echo 'test'")

        assert result["success"] is True
        assert "test" in result["stdout"]
        assert result["exit_code"] == 0

    def test_execute_python(self, executor):
        """Test Python script execution."""
        script = "print('hello from python')"
        result = executor.execute_python(script)

        assert result["success"] is True
        assert "hello from python" in result["stdout"]

    def test_list_files(self, executor):
        """Test listing files."""
        # First create a test file
        executor.execute_bash("echo 'test content' > /workspace/test_list.txt")

        files = executor.list_files("/workspace")

        assert isinstance(files, list)
        assert any("test_list" in f for f in files)

        # Cleanup
        executor.execute_bash("rm -f /workspace/test_list.txt")

    def test_read_write_file(self, executor):
        """Test file read/write operations."""
        test_content = "Hello, this is a test file!"
        test_path = "/workspace/test_rw.txt"

        # Write file
        success = executor.write_file(test_path, test_content)
        assert success is True

        # Read file
        content = executor.read_file(test_path)
        assert content.strip() == test_content

        # Cleanup
        executor.execute_bash(f"rm -f {test_path}")

    def test_command_failure(self, executor):
        """Test handling of failed commands."""
        result = executor.execute_bash("exit 1")

        assert result["success"] is False
        assert result["exit_code"] == 1

    def test_output_truncation(self, executor):
        """Test that large outputs are truncated."""
        # Create executor with small max output
        small_executor = DockerExecutor(
            container_name="manus-sandbox", max_output_length=100
        )

        result = small_executor.execute_bash("seq 1 1000")

        assert "truncated" in result["stdout"]
        assert len(result["stdout"]) < 150  # 100 + truncation message


class TestSeedboxExecutor:
    """Tests for the main SeedboxExecutor facade."""

    def test_default_docker_mode(self):
        """Test that docker mode is used by default."""
        executor = SeedboxExecutor()
        assert executor.mode == "docker"

    def test_execute_simple_command(self):
        """Test simple command execution."""
        executor = SeedboxExecutor()
        result = executor.execute_bash("echo 'hello from seedbox'")

        assert result["success"] is True
        assert "hello from seedbox" in result["stdout"]


# Skip SSH tests unless SSH is configured
class TestSSHExecutor:
    """Tests for SSH-based execution."""

    @pytest.fixture
    def ssh_configured(self):
        """Check if SSH is configured."""
        return bool(os.getenv("SEEDBOX_HOST"))

    def test_ssh_not_configured(self, ssh_configured):
        """Skip SSH tests if not configured."""
        if not ssh_configured:
            pytest.skip("SSH not configured")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
