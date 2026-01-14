"""
Unit tests for FileManagerSkill.

Uses pytest and pytest-asyncio for async test support.
All tests use temporary directories to avoid filesystem side effects.
"""

import json
import pytest
from pathlib import Path

from skills.file_manager import FileManagerSkill, FILE_TYPE_CATEGORIES


class TestFileManagerSkillInit:
    """Tests for FileManagerSkill initialization."""

    def test_default_init(self):
        """Test default initialization."""
        skill = FileManagerSkill()
        assert skill.workspace_dir == Path("/workspace")

    def test_custom_workspace(self):
        """Test custom workspace directory."""
        skill = FileManagerSkill(workspace_dir="/custom/path")
        assert skill.workspace_dir == Path("/custom/path")


class TestValidatePath:
    """Tests for path validation."""

    def test_empty_path_raises(self):
        """Test that empty path raises ValueError."""
        skill = FileManagerSkill()
        with pytest.raises(ValueError, match="Path cannot be empty"):
            skill._validate_path("")

    def test_valid_path(self):
        """Test valid path is resolved."""
        skill = FileManagerSkill()
        result = skill._validate_path("/tmp/test.txt")
        assert result == Path("/tmp/test.txt").resolve()


class TestUploadFile:
    """Tests for upload_file method (local copy)."""

    async def test_upload_success(self, tmp_path):
        """Test successful file copy."""
        skill = FileManagerSkill()

        # Create source file
        source = tmp_path / "source.txt"
        source.write_text("test content")

        dest = tmp_path / "dest" / "copied.txt"

        result = await skill.upload_file(str(source), str(dest))

        assert result["success"] is True
        assert dest.exists()
        assert dest.read_text() == "test content"

    async def test_upload_source_not_found(self, tmp_path):
        """Test error when source doesn't exist."""
        skill = FileManagerSkill()

        with pytest.raises(FileNotFoundError):
            await skill.upload_file(
                str(tmp_path / "nonexistent.txt"), str(tmp_path / "dest.txt")
            )


class TestDownloadFile:
    """Tests for download_file method."""

    async def test_download_empty_url_raises(self):
        """Test that empty URL raises ValueError."""
        skill = FileManagerSkill()
        with pytest.raises(ValueError, match="URL cannot be empty"):
            await skill.download_file("", "/tmp/file.txt")


class TestBatchRename:
    """Tests for batch_rename method."""

    async def test_batch_rename_dry_run(self, tmp_path):
        """Test dry run mode returns planned renames."""
        skill = FileManagerSkill()

        # Create test files
        (tmp_path / "IMG_001.jpg").write_text("")
        (tmp_path / "IMG_002.jpg").write_text("")
        (tmp_path / "other.txt").write_text("")

        results = await skill.batch_rename(
            str(tmp_path), r"IMG_(\d+)", r"photo_\1", dry_run=True
        )

        assert len(results) == 2
        assert all(r["status"] == "dry_run" for r in results)
        assert "photo_001" in results[0]["new_name"]

    async def test_batch_rename_actual(self, tmp_path):
        """Test actual rename operation."""
        skill = FileManagerSkill()

        (tmp_path / "old_file.txt").write_text("content")

        results = await skill.batch_rename(
            str(tmp_path), r"old_", r"new_", dry_run=False
        )

        assert len(results) == 1
        assert results[0]["status"] == "renamed"
        assert (tmp_path / "new_file.txt").exists()
        assert not (tmp_path / "old_file.txt").exists()


class TestConvertFormat:
    """Tests for convert_format method."""

    async def test_csv_to_json(self, tmp_path):
        """Test CSV to JSON conversion."""
        skill = FileManagerSkill()

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25")

        result = await skill.convert_format(str(csv_file), "json")

        json_file = Path(result)
        assert json_file.exists()

        data = json.loads(json_file.read_text())
        assert len(data) == 2
        assert data[0]["name"] == "Alice"
        assert data[1]["age"] == "25"

    async def test_json_to_csv(self, tmp_path):
        """Test JSON to CSV conversion."""
        skill = FileManagerSkill()

        json_file = tmp_path / "data.json"
        json_file.write_text('[{"name": "Alice", "age": 30}]')

        result = await skill.convert_format(str(json_file), "csv")

        csv_file = Path(result)
        assert csv_file.exists()
        content = csv_file.read_text()
        assert "name" in content
        assert "Alice" in content

    async def test_unsupported_conversion_raises(self, tmp_path):
        """Test unsupported conversion raises error."""
        skill = FileManagerSkill()

        txt_file = tmp_path / "file.xyz"
        txt_file.write_text("content")

        with pytest.raises(ValueError, match="Unsupported conversion"):
            await skill.convert_format(str(txt_file), "pdf")


class TestOrganizeByType:
    """Tests for organize_by_type method."""

    async def test_organize_creates_subdirs(self, tmp_path):
        """Test organization creates category subdirectories."""
        skill = FileManagerSkill()

        # Create test files
        (tmp_path / "photo.jpg").write_text("")
        (tmp_path / "document.pdf").write_text("")
        (tmp_path / "data.csv").write_text("")

        result = await skill.organize_by_type(str(tmp_path))

        assert "images" in result
        assert "documents" in result
        assert "spreadsheets" in result

        assert (tmp_path / "images").is_dir()
        assert (tmp_path / "images" / "photo.jpg").exists()


class TestCompressFiles:
    """Tests for compress_files method."""

    async def test_compress_zip(self, tmp_path):
        """Test ZIP compression."""
        skill = FileManagerSkill()

        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        archive_path = tmp_path / "archive.zip"

        result = await skill.compress_files(
            [str(tmp_path / "file1.txt"), str(tmp_path / "file2.txt")],
            str(archive_path),
        )

        assert Path(result).exists()
        assert Path(result).suffix == ".zip"

    async def test_compress_empty_list_raises(self, tmp_path):
        """Test error on empty file list."""
        skill = FileManagerSkill()

        with pytest.raises(ValueError, match="Files list cannot be empty"):
            await skill.compress_files([], str(tmp_path / "archive.zip"))


class TestExtractArchive:
    """Tests for extract_archive method."""

    async def test_extract_zip(self, tmp_path):
        """Test ZIP extraction."""
        skill = FileManagerSkill()

        # Create and compress files
        source = tmp_path / "source"
        source.mkdir()
        (source / "test.txt").write_text("hello")

        archive = tmp_path / "archive.zip"
        await skill.compress_files([str(source / "test.txt")], str(archive))

        # Extract
        dest = tmp_path / "extracted"
        result = await skill.extract_archive(str(archive), str(dest))

        assert len(result) > 0
        assert (dest / "test.txt").exists()


class TestListFiles:
    """Tests for list_files method."""

    async def test_list_files_basic(self, tmp_path):
        """Test basic file listing."""
        skill = FileManagerSkill()

        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.py").write_text("b")
        (tmp_path / "subdir").mkdir()

        files = await skill.list_files(str(tmp_path))

        assert len(files) == 2
        assert all("path" in f for f in files)
        assert all("size_bytes" in f for f in files)

    async def test_list_files_with_pattern(self, tmp_path):
        """Test file listing with glob pattern."""
        skill = FileManagerSkill()

        (tmp_path / "test.py").write_text("")
        (tmp_path / "other.txt").write_text("")

        files = await skill.list_files(str(tmp_path), pattern="*.py")

        assert len(files) == 1
        assert files[0]["extension"] == ".py"


class TestReadWriteFile:
    """Tests for read_file and write_file methods."""

    async def test_read_write_roundtrip(self, tmp_path):
        """Test write then read returns same content."""
        skill = FileManagerSkill()

        content = "Hello, World!\nLine 2\nÜnicode: 日本語"

        result = await skill.write_file(str(tmp_path / "test.txt"), content)
        assert result["success"] is True

        read_content = await skill.read_file(str(tmp_path / "test.txt"))
        assert read_content == content


class TestDeleteFile:
    """Tests for delete_file method."""

    async def test_delete_existing(self, tmp_path):
        """Test deleting existing file."""
        skill = FileManagerSkill()

        file_path = tmp_path / "to_delete.txt"
        file_path.write_text("delete me")

        result = await skill.delete_file(str(file_path))

        assert result is True
        assert not file_path.exists()

    async def test_delete_nonexistent(self, tmp_path):
        """Test deleting non-existent file returns False."""
        skill = FileManagerSkill()

        result = await skill.delete_file(str(tmp_path / "nonexistent.txt"))

        assert result is False
