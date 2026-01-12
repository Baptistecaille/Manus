"""
SeedboxExecutor - Communication module for isolated sandbox execution.

Supports two modes:
- Docker: Execute commands via docker exec
- SSH: Execute commands via Paramiko SSH connection

All methods return a standardized dict: {success, stdout, stderr, error?}
"""

import logging
import os
import subprocess
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    import paramiko
except ImportError:
    paramiko = None  # type: ignore

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionResult:
    """Standardized execution result."""

    def __init__(
        self,
        success: bool,
        stdout: str = "",
        stderr: str = "",
        error: Optional[str] = None,
        exit_code: int = -1,
    ):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.exit_code = exit_code

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        result = {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
        }
        if self.error:
            result["error"] = self.error
        return result


class BaseExecutor(ABC):
    """Abstract base class for sandbox executors."""

    def __init__(self, timeout: int = 30, max_output_length: int = 5000):
        """
        Initialize the executor.

        Args:
            timeout: Maximum execution time in seconds.
            max_output_length: Maximum length of stdout/stderr to return.
        """
        self.timeout = timeout
        self.max_output_length = max_output_length

    def _truncate_output(self, output: str) -> str:
        """Truncate output if it exceeds max length."""
        if len(output) > self.max_output_length:
            return (
                output[: self.max_output_length]
                + f"\n... [truncated, {len(output) - self.max_output_length} more chars]"
            )
        return output

    @abstractmethod
    def execute_bash(self, command: str) -> dict:
        """Execute a bash command in the sandbox."""
        pass

    @abstractmethod
    def execute_python(self, script: str) -> dict:
        """Execute a Python script in the sandbox."""
        pass

    @abstractmethod
    def list_files(self, directory: str = "/workspace") -> list[str]:
        """List files in a directory."""
        pass

    @abstractmethod
    def read_file(self, filepath: str) -> str:
        """Read content of a file."""
        pass

    @abstractmethod
    def write_file(self, filepath: str, content: str) -> bool:
        """Write content to a file."""
        pass


class DockerExecutor(BaseExecutor):
    """Execute commands in a Docker container."""

    def __init__(
        self,
        container_name: str = "manus-sandbox",
        timeout: int = 30,
        max_output_length: int = 5000,
    ):
        """
        Initialize Docker executor.

        Args:
            container_name: Name of the Docker container.
            timeout: Maximum execution time in seconds.
            max_output_length: Maximum length of stdout/stderr to return.
        """
        super().__init__(timeout, max_output_length)
        self.container_name = container_name
        logger.info(f"Initialized DockerExecutor for container: {container_name}")

    def _docker_exec(self, command: str, shell: bool = True) -> ExecutionResult:
        """Execute a command in the Docker container."""
        try:
            if shell:
                docker_cmd = [
                    "docker",
                    "exec",
                    self.container_name,
                    "bash",
                    "-c",
                    command,
                ]
            else:
                docker_cmd = ["docker", "exec", self.container_name] + shlex.split(
                    command
                )

            logger.debug(f"Executing: {' '.join(docker_cmd)}")

            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=self.timeout
            )

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=self._truncate_output(result.stdout),
                stderr=self._truncate_output(result.stderr),
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out after {self.timeout}s: {command[:100]}")
            return ExecutionResult(
                success=False, error=f"Command timed out after {self.timeout} seconds"
            )
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error: {e}")
            return ExecutionResult(success=False, error=f"Subprocess error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ExecutionResult(success=False, error=f"Unexpected error: {str(e)}")

    def execute_bash(self, command: str) -> dict:
        """
        Execute a bash command in the Docker container.

        Args:
            command: The bash command to execute.

        Returns:
            Dict with success, stdout, stderr, exit_code, and optional error.
        """
        logger.info(f"Executing bash: {command[:100]}...")
        result = self._docker_exec(command)
        return result.to_dict()

    def execute_python(self, script: str) -> dict:
        """
        Execute a Python script in the Docker container.

        Args:
            script: The Python code to execute.

        Returns:
            Dict with success, stdout, stderr, exit_code, and optional error.
        """
        logger.info(f"Executing Python script ({len(script)} chars)")
        # Escape the script for shell execution
        escaped_script = script.replace("'", "'\"'\"'")
        command = f"python3 -c '{escaped_script}'"
        result = self._docker_exec(command)
        return result.to_dict()

    def list_files(self, directory: str = "/workspace") -> list[str]:
        """
        List files in a directory within the container.

        Args:
            directory: Directory path to list.

        Returns:
            List of file paths, or empty list on error.
        """
        result = self._docker_exec(
            f"find {shlex.quote(directory)} -maxdepth 2 -type f 2>/dev/null | head -100"
        )
        if result.success and result.stdout:
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return []

    def read_file(self, filepath: str) -> str:
        """
        Read content of a file in the container.

        Args:
            filepath: Path to the file to read.

        Returns:
            File content, or empty string on error.
        """
        result = self._docker_exec(f"cat {shlex.quote(filepath)}")
        if result.success:
            return result.stdout
        logger.warning(
            f"Failed to read file {filepath}: {result.error or result.stderr}"
        )
        return ""

    def write_file(self, filepath: str, content: str) -> bool:
        """
        Write content to a file in the container.

        Args:
            filepath: Path to the file to write.
            content: Content to write.

        Returns:
            True if successful, False otherwise.
        """
        # Use base64 encoding to safely pass content
        import base64

        encoded = base64.b64encode(content.encode()).decode()
        command = f"echo '{encoded}' | base64 -d > {shlex.quote(filepath)}"
        result = self._docker_exec(command)
        return result.success


class SSHExecutor(BaseExecutor):
    """Execute commands via SSH connection."""

    def __init__(
        self,
        host: str,
        username: str,
        port: int = 22,
        key_path: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
        max_output_length: int = 5000,
    ):
        """
        Initialize SSH executor.

        Args:
            host: SSH host address.
            username: SSH username.
            port: SSH port (default 22).
            key_path: Path to SSH private key.
            password: SSH password (if not using key).
            timeout: Maximum execution time in seconds.
            max_output_length: Maximum length of stdout/stderr to return.
        """
        super().__init__(timeout, max_output_length)

        if paramiko is None:
            raise ImportError(
                "paramiko is required for SSH mode. Install with: pip install paramiko"
            )

        self.host = host
        self.username = username
        self.port = port
        self.key_path = key_path
        self.password = password
        self._client: Optional[paramiko.SSHClient] = None

        logger.info(f"Initialized SSHExecutor for {username}@{host}:{port}")

    def _get_client(self) -> paramiko.SSHClient:
        """Get or create SSH client connection."""
        if self._client is None:
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": self.host,
                "port": self.port,
                "username": self.username,
                "timeout": self.timeout,
            }

            if self.key_path:
                connect_kwargs["key_filename"] = self.key_path
            elif self.password:
                connect_kwargs["password"] = self.password

            self._client.connect(**connect_kwargs)
            logger.info("SSH connection established")

        return self._client

    def _ssh_exec(self, command: str) -> ExecutionResult:
        """Execute a command via SSH."""
        try:
            client = self._get_client()
            stdin, stdout, stderr = client.exec_command(command, timeout=self.timeout)

            exit_code = stdout.channel.recv_exit_status()
            stdout_str = self._truncate_output(stdout.read().decode())
            stderr_str = self._truncate_output(stderr.read().decode())

            return ExecutionResult(
                success=exit_code == 0,
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
            )

        except Exception as e:
            logger.error(f"SSH execution error: {e}")
            # Reset client on error
            self._client = None
            return ExecutionResult(success=False, error=f"SSH error: {str(e)}")

    def execute_bash(self, command: str) -> dict:
        """Execute a bash command via SSH."""
        logger.info(f"Executing bash via SSH: {command[:100]}...")
        result = self._ssh_exec(command)
        return result.to_dict()

    def execute_python(self, script: str) -> dict:
        """Execute a Python script via SSH."""
        logger.info(f"Executing Python script via SSH ({len(script)} chars)")
        escaped_script = script.replace("'", "'\"'\"'")
        command = f"python3 -c '{escaped_script}'"
        result = self._ssh_exec(command)
        return result.to_dict()

    def list_files(self, directory: str = "/workspace") -> list[str]:
        """List files in a directory via SSH."""
        result = self._ssh_exec(
            f"find {shlex.quote(directory)} -maxdepth 2 -type f 2>/dev/null | head -100"
        )
        if result.success and result.stdout:
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return []

    def read_file(self, filepath: str) -> str:
        """Read content of a file via SSH."""
        result = self._ssh_exec(f"cat {shlex.quote(filepath)}")
        if result.success:
            return result.stdout
        return ""

    def write_file(self, filepath: str, content: str) -> bool:
        """Write content to a file via SSH."""
        import base64

        encoded = base64.b64encode(content.encode()).decode()
        command = f"echo '{encoded}' | base64 -d > {shlex.quote(filepath)}"
        result = self._ssh_exec(command)
        return result.success

    def close(self):
        """Close SSH connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("SSH connection closed")


class SeedboxExecutor:
    """
    Factory class for creating the appropriate executor based on configuration.

    Usage:
        executor = SeedboxExecutor()
        result = executor.execute_bash("echo 'Hello World'")
        print(result)  # {'success': True, 'stdout': 'Hello World\n', ...}
    """

    def __init__(self, mode: Optional[str] = None, **kwargs):
        """
        Initialize SeedboxExecutor.

        Args:
            mode: Execution mode ('docker' or 'ssh'). Defaults to SEEDBOX_MODE env var.
            **kwargs: Additional arguments passed to the underlying executor.
        """
        self.mode = mode or os.getenv("SEEDBOX_MODE", "docker")

        timeout = int(os.getenv("EXECUTION_TIMEOUT", "30"))
        max_output_length = int(os.getenv("MAX_OUTPUT_LENGTH", "5000"))

        if self.mode == "docker":
            container_name = kwargs.get(
                "container_name", os.getenv("DOCKER_CONTAINER_NAME", "manus-sandbox")
            )
            self._executor = DockerExecutor(
                container_name=container_name,
                timeout=timeout,
                max_output_length=max_output_length,
            )
        elif self.mode == "ssh":
            self._executor = SSHExecutor(
                host=kwargs.get("host", os.getenv("SEEDBOX_HOST", "")),
                username=kwargs.get("username", os.getenv("SEEDBOX_USER", "")),
                port=int(kwargs.get("port", os.getenv("SEEDBOX_PORT", "22"))),
                key_path=kwargs.get("key_path", os.getenv("SEEDBOX_SSH_KEY_PATH")),
                password=kwargs.get("password", os.getenv("SEEDBOX_PASSWORD")),
                timeout=timeout,
                max_output_length=max_output_length,
            )
        else:
            raise ValueError(
                f"Unknown seedbox mode: {self.mode}. Use 'docker' or 'ssh'."
            )

        logger.info(f"SeedboxExecutor initialized in {self.mode} mode")

    def execute_bash(self, command: str) -> dict:
        """Execute a bash command in the sandbox."""
        return self._executor.execute_bash(command)

    def execute_python(self, script: str) -> dict:
        """Execute a Python script in the sandbox."""
        return self._executor.execute_python(script)

    def list_files(self, directory: str = "/workspace") -> list[str]:
        """List files in a directory."""
        return self._executor.list_files(directory)

    def read_file(self, filepath: str) -> str:
        """Read content of a file."""
        return self._executor.read_file(filepath)

    def write_file(self, filepath: str, content: str) -> bool:
        """Write content to a file."""
        return self._executor.write_file(filepath, content)

    def close(self):
        """Close any open connections."""
        if hasattr(self._executor, "close"):
            self._executor.close()


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    executor = SeedboxExecutor()
    result = executor.execute_bash("echo 'Hello from Seedbox!'")
    print(f"Result: {result}")
