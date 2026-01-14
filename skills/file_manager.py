"""
File Manager Skill - Advanced file management for Manus agent.

Provides async file operations including upload/download, batch operations,
format conversion, compression, and automatic file organization.
"""

import asyncio
import csv
import io
import json
import logging
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import aiofiles
import aiohttp

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

# File type categories for organization
FILE_TYPE_CATEGORIES: dict[str, list[str]] = {
    "documents": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".md"],
    "spreadsheets": [".csv", ".xls", ".xlsx", ".ods"],
    "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico"],
    "videos": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"],
    "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma"],
    "archives": [".zip", ".tar", ".gz", ".rar", ".7z", ".tar.gz", ".tgz"],
    "code": [".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs"],
    "data": [".json", ".xml", ".yaml", ".yml", ".toml"],
}


class FileManagerSkill:
    """
    Skill for advanced file management operations.

    Provides async methods for file download, upload, batch rename,
    format conversion, compression, and organization.

    Attributes:
        workspace_dir: Default working directory for file operations.

    Example:
        >>> skill = FileManagerSkill()
        >>> await skill.download_file("https://example.com/file.pdf", "/tmp/file.pdf")
        >>> result = await skill.organize_by_type("/tmp/downloads")
    """

    def __init__(self, workspace_dir: Optional[str] = None) -> None:
        """
        Initialize the file manager skill.

        Args:
            workspace_dir: Default workspace directory. Defaults to /workspace.
        """
        self.workspace_dir = Path(
            workspace_dir or os.getenv("WORKSPACE_DIR", "/workspace")
        )
        logger.debug(f"FileManagerSkill initialized (workspace={self.workspace_dir})")

    def _validate_path(self, path: str | Path) -> Path:
        """
        Validate and resolve a file path.

        Args:
            path: Path to validate.

        Returns:
            Resolved Path object.

        Raises:
            ValueError: If path is empty or invalid.
        """
        if not path:
            raise ValueError("Path cannot be empty")

        resolved = Path(path).resolve()
        return resolved

    async def upload_file(
        self,
        local_path: str,
        destination: str,
    ) -> dict[str, Any]:
        """
        Copy a file from local path to destination.

        For local-only operations (no remote upload support).

        Args:
            local_path: Source file path.
            destination: Target destination path.

        Returns:
            Dict with success status and file info.

        Raises:
            ValueError: If paths are invalid.
            FileNotFoundError: If source file doesn't exist.
        """
        source = self._validate_path(local_path)
        dest = self._validate_path(destination)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        logger.info(f"Copying file: {source} → {dest}")

        try:
            # Ensure destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Use async copy
            async with aiofiles.open(source, "rb") as src:
                content = await src.read()
            async with aiofiles.open(dest, "wb") as dst:
                await dst.write(content)

            return {
                "success": True,
                "source": str(source),
                "destination": str(dest),
                "size_bytes": dest.stat().st_size,
            }

        except Exception as e:
            logger.error(f"File copy failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": str(source),
                "destination": str(dest),
            }

    async def download_file(
        self,
        url: str,
        destination: str,
        timeout: int = 60,
    ) -> str:
        """
        Download a file from URL to local destination.

        Args:
            url: URL to download from.
            destination: Local path to save file.
            timeout: Request timeout in seconds.

        Returns:
            Path to downloaded file.

        Raises:
            ValueError: If URL or destination is invalid.
            RuntimeError: If download fails.

        Example:
            >>> path = await skill.download_file(
            ...     "https://example.com/data.csv",
            ...     "/tmp/data.csv"
            ... )
        """
        if not url:
            raise ValueError("URL cannot be empty")

        dest = self._validate_path(destination)
        logger.info(f"Downloading: {url} → {dest}")

        try:
            # Ensure destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Download failed: HTTP {response.status}")

                    async with aiofiles.open(dest, "wb") as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)

            logger.info(f"Download complete: {dest} ({dest.stat().st_size} bytes)")
            return str(dest)

        except asyncio.TimeoutError:
            raise RuntimeError(f"Download timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise RuntimeError(f"Download failed: {e}") from e

    async def batch_rename(
        self,
        directory: str,
        pattern: str,
        replacement: str,
        recursive: bool = False,
        dry_run: bool = False,
    ) -> list[dict[str, str]]:
        """
        Batch rename files matching a pattern.

        Args:
            directory: Directory to search for files.
            pattern: Regex pattern to match in filenames.
            replacement: Replacement string (supports regex groups).
            recursive: Include subdirectories.
            dry_run: If True, only return what would be renamed.

        Returns:
            List of dicts with old and new names.

        Example:
            >>> results = await skill.batch_rename(
            ...     "/tmp/photos",
            ...     r"IMG_(\d+)",
            ...     r"photo_\1"
            ... )
        """
        dir_path = self._validate_path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        logger.info(f"Batch rename in {dir_path} (pattern={pattern})")

        regex = re.compile(pattern)
        results: list[dict[str, str]] = []

        # Get files
        if recursive:
            files = list(dir_path.rglob("*"))
        else:
            files = list(dir_path.iterdir())

        for file_path in files:
            if not file_path.is_file():
                continue

            old_name = file_path.name
            new_name = regex.sub(replacement, old_name)

            if new_name != old_name:
                new_path = file_path.parent / new_name
                result = {
                    "old_path": str(file_path),
                    "new_path": str(new_path),
                    "old_name": old_name,
                    "new_name": new_name,
                }

                if not dry_run:
                    try:
                        file_path.rename(new_path)
                        result["status"] = "renamed"
                    except Exception as e:
                        result["status"] = "failed"
                        result["error"] = str(e)
                else:
                    result["status"] = "dry_run"

                results.append(result)

        logger.info(f"Batch rename complete: {len(results)} files processed")
        return results

    async def convert_format(
        self,
        file_path: str,
        target_format: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Convert file between formats (CSV ↔ JSON supported).

        Args:
            file_path: Source file path.
            target_format: Target format ('csv', 'json', 'txt').
            output_path: Optional output path. If None, uses same dir with new extension.

        Returns:
            Path to converted file.

        Raises:
            ValueError: If conversion not supported.

        Example:
            >>> json_path = await skill.convert_format("/tmp/data.csv", "json")
        """
        source = self._validate_path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        source_ext = source.suffix.lower()
        target_ext = f".{target_format.lower().lstrip('.')}"

        if output_path:
            dest = self._validate_path(output_path)
        else:
            dest = source.with_suffix(target_ext)

        logger.info(f"Converting: {source} → {dest}")

        # CSV to JSON
        if source_ext == ".csv" and target_ext == ".json":
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                content = await f.read()
            reader = csv.DictReader(io.StringIO(content))
            data = list(reader)
            async with aiofiles.open(dest, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

        # JSON to CSV
        elif source_ext == ".json" and target_ext == ".csv":
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
            if not isinstance(data, list) or not data:
                raise ValueError("JSON must be a non-empty array of objects")

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            async with aiofiles.open(dest, "w", encoding="utf-8") as f:
                await f.write(output.getvalue())

        # To TXT (any format)
        elif target_ext == ".txt":
            async with aiofiles.open(source, "r", encoding="utf-8") as f:
                content = await f.read()
            # For JSON, pretty print
            if source_ext == ".json":
                data = json.loads(content)
                content = json.dumps(data, indent=2, ensure_ascii=False)
            async with aiofiles.open(dest, "w", encoding="utf-8") as f:
                await f.write(content)

        else:
            raise ValueError(f"Unsupported conversion: {source_ext} → {target_ext}")

        logger.info(f"Conversion complete: {dest}")
        return str(dest)

    async def organize_by_type(
        self,
        directory: str,
        create_subdirs: bool = True,
    ) -> dict[str, list[str]]:
        """
        Organize files in directory by type into subdirectories.

        Args:
            directory: Directory to organize.
            create_subdirs: If True, create category subdirectories and move files.

        Returns:
            Dict mapping categories to list of files.

        Example:
            >>> result = await skill.organize_by_type("/tmp/downloads")
            >>> print(result["images"])  # List of image files
        """
        dir_path = self._validate_path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        logger.info(f"Organizing directory: {dir_path}")

        organized: dict[str, list[str]] = {cat: [] for cat in FILE_TYPE_CATEGORIES}
        organized["other"] = []

        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            category = "other"

            for cat, extensions in FILE_TYPE_CATEGORIES.items():
                if ext in extensions:
                    category = cat
                    break

            if create_subdirs:
                target_dir = dir_path / category
                target_dir.mkdir(exist_ok=True)
                target_path = target_dir / file_path.name

                try:
                    shutil.move(str(file_path), str(target_path))
                    organized[category].append(str(target_path))
                except Exception as e:
                    logger.warning(f"Failed to move {file_path}: {e}")
                    organized[category].append(str(file_path))
            else:
                organized[category].append(str(file_path))

        # Remove empty categories
        organized = {k: v for k, v in organized.items() if v}

        logger.info(
            f"Organization complete: {sum(len(v) for v in organized.values())} files"
        )
        return organized

    async def compress_files(
        self,
        files: list[str],
        output_path: str,
        format: str = "zip",
    ) -> str:
        """
        Compress files into an archive.

        Args:
            files: List of file paths to compress.
            output_path: Output archive path.
            format: Archive format ('zip' or 'tar.gz').

        Returns:
            Path to created archive.

        Example:
            >>> path = await skill.compress_files(
            ...     ["/tmp/file1.txt", "/tmp/file2.txt"],
            ...     "/tmp/archive.zip"
            ... )
        """
        if not files:
            raise ValueError("Files list cannot be empty")

        dest = self._validate_path(output_path)
        logger.info(f"Compressing {len(files)} files to: {dest}")

        # Validate all source files exist
        paths = [self._validate_path(f) for f in files]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")

        dest.parent.mkdir(parents=True, exist_ok=True)

        if format == "zip":
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in paths:
                    zf.write(path, path.name)

        elif format in ("tar.gz", "tgz"):
            with tarfile.open(dest, "w:gz") as tf:
                for path in paths:
                    tf.add(path, arcname=path.name)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Archive created: {dest} ({dest.stat().st_size} bytes)")
        return str(dest)

    async def extract_archive(
        self,
        archive_path: str,
        destination: str,
    ) -> list[str]:
        """
        Extract files from an archive.

        Args:
            archive_path: Path to archive file.
            destination: Directory to extract to.

        Returns:
            List of extracted file paths.

        Example:
            >>> files = await skill.extract_archive(
            ...     "/tmp/archive.zip",
            ...     "/tmp/extracted"
            ... )
        """
        archive = self._validate_path(archive_path)
        dest_dir = self._validate_path(destination)

        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")

        logger.info(f"Extracting: {archive} → {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        extracted: list[str] = []
        ext = archive.suffix.lower()

        if ext == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest_dir)
                extracted = [str(dest_dir / name) for name in zf.namelist()]

        elif ext in (".gz", ".tgz") or str(archive).endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(dest_dir)
                extracted = [str(dest_dir / member.name) for member in tf.getmembers()]

        elif ext == ".tar":
            with tarfile.open(archive, "r") as tf:
                tf.extractall(dest_dir)
                extracted = [str(dest_dir / member.name) for member in tf.getmembers()]

        else:
            raise ValueError(f"Unsupported archive format: {ext}")

        logger.info(f"Extracted {len(extracted)} items")
        return extracted

    async def list_files(
        self,
        directory: str,
        pattern: Optional[str] = None,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        """
        List files in a directory with details.

        Args:
            directory: Directory to list.
            pattern: Optional glob pattern to filter.
            recursive: Include subdirectories.

        Returns:
            List of file info dicts.
        """
        dir_path = self._validate_path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        files: list[dict[str, Any]] = []

        if recursive:
            iterator = dir_path.rglob(pattern or "*")
        else:
            iterator = dir_path.glob(pattern or "*")

        for path in iterator:
            if path.is_file():
                stat = path.stat()
                files.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": stat.st_size,
                        "extension": path.suffix.lower(),
                        "modified": stat.st_mtime,
                    }
                )

        return files

    async def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> str:
        """
        Read text content from a file.

        Args:
            file_path: Path to file.
            encoding: Text encoding.

        Returns:
            File content as string.
        """
        path = self._validate_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        async with aiofiles.open(path, "r", encoding=encoding) as f:
            return await f.read()

    async def write_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """
        Write text content to a file.

        Args:
            file_path: Path to file.
            content: Content to write.
            encoding: Text encoding.

        Returns:
            Dict with file info.
        """
        path = self._validate_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(path, "w", encoding=encoding) as f:
            await f.write(content)

        return {
            "success": True,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.

        Args:
            file_path: Path to file to delete.

        Returns:
            True if deleted successfully.
        """
        path = self._validate_path(file_path)
        if not path.exists():
            return False

        path.unlink()
        logger.info(f"Deleted: {path}")
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    async def test_file_manager():
        """Quick test of file manager capabilities."""
        print("=== File Manager Skill Test ===\n")

        skill = FileManagerSkill()
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test write file
            print("1. Testing write file...")
            result = await skill.write_file(
                f"{tmpdir}/test.txt", "Hello, World!\nLine 2"
            )
            print(f"   Written: {result['size_bytes']} bytes")

            # Test read file
            print("\n2. Testing read file...")
            content = await skill.read_file(f"{tmpdir}/test.txt")
            print(f"   Read: {len(content)} chars")

            # Test CSV to JSON conversion
            print("\n3. Testing format conversion...")
            await skill.write_file(f"{tmpdir}/data.csv", "name,age\nAlice,30\nBob,25")
            json_path = await skill.convert_format(f"{tmpdir}/data.csv", "json")
            print(f"   Converted: {json_path}")

            # Test list files
            print("\n4. Testing list files...")
            files = await skill.list_files(tmpdir)
            print(f"   Found: {len(files)} files")

            # Test compress
            print("\n5. Testing compression...")
            archive = await skill.compress_files(
                [f"{tmpdir}/test.txt", f"{tmpdir}/data.csv"], f"{tmpdir}/archive.zip"
            )
            print(f"   Archive: {archive}")

            # Test extract
            print("\n6. Testing extraction...")
            extracted = await skill.extract_archive(archive, f"{tmpdir}/extracted")
            print(f"   Extracted: {len(extracted)} files")

        print("\n=== Test Complete ===")

    asyncio.run(test_file_manager())
