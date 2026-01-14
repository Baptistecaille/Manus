"""
Unit tests for DataAnalyzerSkill.

Uses pytest and pytest-asyncio for async test support.
All tests use temporary files to avoid filesystem side effects.
"""

import json
import pytest
from pathlib import Path

from skills.data_analyzer import DataAnalyzerSkill


class TestDataAnalyzerInit:
    """Tests for DataAnalyzerSkill initialization."""

    def test_default_init(self):
        """Test default initialization."""
        skill = DataAnalyzerSkill()
        assert skill.workspace_dir == Path("/workspace")

    def test_custom_workspace(self, tmp_path):
        """Test custom workspace directory."""
        skill = DataAnalyzerSkill(str(tmp_path))
        assert skill.workspace_dir == tmp_path


class TestAnalyzeCSV:
    """Tests for analyze_csv method."""

    async def test_analyze_simple_csv(self, tmp_path):
        """Test analyzing a simple CSV file."""
        skill = DataAnalyzerSkill(str(tmp_path))

        # Create test CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,score\nAlice,30,85\nBob,25,90\n")

        result = await skill.analyze_csv(str(csv_file))

        assert result["shape"] == (2, 3)
        assert "name" in result["columns"]
        assert "age" in result["columns"]
        assert result["null_counts"]["name"] == 0

    async def test_analyze_csv_not_found(self, tmp_path):
        """Test error on missing file."""
        skill = DataAnalyzerSkill(str(tmp_path))

        with pytest.raises(FileNotFoundError):
            await skill.analyze_csv(str(tmp_path / "missing.csv"))


class TestAnalyzeJSON:
    """Tests for analyze_json method."""

    async def test_analyze_json_array(self, tmp_path):
        """Test analyzing a JSON array."""
        skill = DataAnalyzerSkill(str(tmp_path))

        json_file = tmp_path / "test.json"
        json_file.write_text('[{"name": "Alice"}, {"name": "Bob"}]')

        result = await skill.analyze_json(str(json_file))

        assert result["type"] == "array"
        assert result["length"] == 2

    async def test_analyze_json_object(self, tmp_path):
        """Test analyzing a JSON object."""
        skill = DataAnalyzerSkill(str(tmp_path))

        json_file = tmp_path / "test.json"
        json_file.write_text('{"key1": "value1", "key2": "value2"}')

        result = await skill.analyze_json(str(json_file))

        assert result["type"] == "object"
        assert result["key_count"] == 2


class TestGenerateChart:
    """Tests for generate_chart method."""

    async def test_generate_bar_chart(self, tmp_path):
        """Test generating a bar chart."""
        skill = DataAnalyzerSkill(str(tmp_path))

        # Create test CSV
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("category,value\nA,10\nB,20\nC,15\n")

        output = tmp_path / "chart.png"

        result = await skill.generate_chart(
            str(csv_file),
            chart_type="bar",
            output=str(output),
            x_column="category",
            y_column="value",
        )

        assert Path(result).exists()
        assert Path(result).suffix == ".png"

    async def test_generate_line_chart(self, tmp_path):
        """Test generating a line chart."""
        skill = DataAnalyzerSkill(str(tmp_path))

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("x,y\n1,5\n2,10\n3,8\n")

        output = tmp_path / "line.png"

        result = await skill.generate_chart(
            str(csv_file),
            chart_type="line",
            output=str(output),
        )

        assert Path(result).exists()


class TestCreateReport:
    """Tests for create_report method."""

    async def test_markdown_report(self, tmp_path):
        """Test creating a markdown report."""
        skill = DataAnalyzerSkill(str(tmp_path))

        analysis = {
            "shape": (100, 5),
            "columns": ["id", "name", "value"],
            "null_counts": {"id": 0, "name": 2, "value": 0},
        }

        report = await skill.create_report(analysis, format="markdown")

        assert "# Data Analysis Report" in report
        assert "100" in report
        assert "name" in report

    async def test_report_to_file(self, tmp_path):
        """Test saving report to file."""
        skill = DataAnalyzerSkill(str(tmp_path))

        analysis = {"shape": (10, 2), "columns": ["a", "b"]}
        output = tmp_path / "report.md"

        await skill.create_report(analysis, output=str(output))

        assert output.exists()
        assert "Data Analysis Report" in output.read_text()
