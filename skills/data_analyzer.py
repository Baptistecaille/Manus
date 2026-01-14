"""
Data Analyzer Skill - Statistical analysis and visualization for Manus agent.

Provides async methods for CSV/JSON analysis, chart generation, and report creation
using pandas and matplotlib.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Literal

# Lazy imports to avoid startup overhead
_pd = None
_plt = None

logger = logging.getLogger(__name__)


def _get_pandas():
    """Lazy import pandas."""
    global _pd
    if _pd is None:
        import pandas as pd

        _pd = pd
    return _pd


def _get_matplotlib():
    """Lazy import matplotlib."""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt

        # Use non-interactive backend for server use
        import matplotlib

        matplotlib.use("Agg")
        _plt = plt
    return _plt


class DataAnalyzerSkill:
    """
    Skill for data analysis and visualization.

    Provides async methods for analyzing CSV/JSON data, generating charts,
    and creating formatted reports.

    Attributes:
        workspace_dir: Directory for saving outputs.

    Example:
        >>> analyzer = DataAnalyzerSkill()
        >>> stats = await analyzer.analyze_csv("/data/sales.csv")
        >>> chart = await analyzer.generate_chart(
        ...     "/data/sales.csv",
        ...     chart_type="bar",
        ...     output="/output/chart.png"
        ... )
    """

    def __init__(self, workspace_dir: str = "/workspace"):
        """
        Initialize the data analyzer skill.

        Args:
            workspace_dir: Directory for saving outputs.
        """
        self.workspace_dir = Path(workspace_dir)

    async def analyze_csv(
        self,
        file_path: str,
        sample_size: int = 1000,
    ) -> dict[str, Any]:
        """
        Analyze a CSV file and return statistics.

        Args:
            file_path: Path to CSV file.
            sample_size: Max rows to load for large files.

        Returns:
            Dict with shape, columns, summary stats, null counts, dtypes.

        Example:
            >>> stats = await analyzer.analyze_csv("/data/users.csv")
            >>> print(stats["shape"])  # (1000, 5)
            >>> print(stats["columns"])  # ["id", "name", "age", ...]
        """
        pd = _get_pandas()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load with sampling for large files
        loop = asyncio.get_event_loop()

        def _load_and_analyze():
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 100:
                df = pd.read_csv(path, nrows=sample_size)
                sampled = True
            else:
                df = pd.read_csv(path)
                sampled = False

            return {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "summary": df.describe(include="all").to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "sample_rows": df.head(5).to_dict(orient="records"),
                "sampled": sampled,
                "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            }

        result = await loop.run_in_executor(None, _load_and_analyze)
        logger.info(f"Analyzed CSV: {file_path}, shape: {result['shape']}")
        return result

    async def analyze_json(
        self,
        file_path: str,
    ) -> dict[str, Any]:
        """
        Analyze a JSON file and return structure information.

        Args:
            file_path: Path to JSON file.

        Returns:
            Dict with structure analysis.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        loop = asyncio.get_event_loop()

        def _load_and_analyze():
            with open(path) as f:
                data = json.load(f)

            if isinstance(data, list):
                return {
                    "type": "array",
                    "length": len(data),
                    "sample": data[:3] if data else [],
                    "item_type": type(data[0]).__name__ if data else None,
                }
            elif isinstance(data, dict):
                return {
                    "type": "object",
                    "keys": list(data.keys())[:20],
                    "key_count": len(data),
                    "sample": {k: v for k, v in list(data.items())[:3]},
                }
            else:
                return {
                    "type": type(data).__name__,
                    "value": str(data)[:100],
                }

        return await loop.run_in_executor(None, _load_and_analyze)

    async def generate_chart(
        self,
        data_source: str,
        chart_type: Literal["line", "bar", "pie", "scatter", "histogram"],
        output: str,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple[int, int] = (10, 6),
    ) -> str:
        """
        Generate a chart from data.

        Args:
            data_source: Path to CSV/JSON file or inline JSON data.
            chart_type: Type of chart to generate.
            output: Output file path for the chart image.
            x_column: Column for x-axis (auto-detected if not provided).
            y_column: Column for y-axis (auto-detected if not provided).
            title: Chart title.
            figsize: Figure size in inches (width, height).

        Returns:
            Path to the generated chart image.

        Example:
            >>> path = await analyzer.generate_chart(
            ...     "/data/sales.csv",
            ...     chart_type="bar",
            ...     output="/output/sales_chart.png",
            ...     x_column="month",
            ...     y_column="revenue"
            ... )
        """
        pd = _get_pandas()
        plt = _get_matplotlib()

        loop = asyncio.get_event_loop()

        def _generate():
            # Load data
            source_path = Path(data_source)
            if source_path.exists():
                if source_path.suffix == ".csv":
                    df = pd.read_csv(source_path)
                elif source_path.suffix == ".json":
                    df = pd.read_json(source_path)
                else:
                    raise ValueError(f"Unsupported file type: {source_path.suffix}")
            else:
                # Try parsing as inline JSON
                df = pd.read_json(StringIO(data_source))

            # Auto-detect columns if not provided
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if x_column is None:
                x = df.index if len(numeric_cols) < 2 else df.iloc[:, 0]
                x_label = "Index"
            else:
                x = df[x_column]
                x_label = x_column

            if y_column is None:
                y_col = numeric_cols[0] if numeric_cols else df.columns[0]
                y = df[y_col]
                y_label = y_col
            else:
                y = df[y_column]
                y_label = y_column

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            if chart_type == "line":
                ax.plot(x, y, marker="o", linewidth=2)
            elif chart_type == "bar":
                ax.bar(range(len(y)), y, tick_label=x if len(x) <= 20 else None)
            elif chart_type == "pie":
                ax.pie(y, labels=x if len(x) <= 10 else None, autopct="%1.1f%%")
            elif chart_type == "scatter":
                ax.scatter(x, y, alpha=0.7)
            elif chart_type == "histogram":
                ax.hist(y, bins=30, edgecolor="black", alpha=0.7)

            if chart_type != "pie":
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

            ax.set_title(title or f"{chart_type.title()} Chart: {y_label}")
            plt.tight_layout()

            # Save
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            return str(output_path.absolute())

        return await loop.run_in_executor(None, _generate)

    async def create_report(
        self,
        analysis: dict[str, Any],
        format: Literal["markdown", "html", "json"] = "markdown",
        output: Optional[str] = None,
    ) -> str:
        """
        Create a formatted report from analysis results.

        Args:
            analysis: Analysis dict from analyze_csv/analyze_json.
            format: Output format.
            output: Optional output file path.

        Returns:
            Formatted report as string.

        Example:
            >>> stats = await analyzer.analyze_csv("/data.csv")
            >>> report = await analyzer.create_report(stats, format="markdown")
        """
        timestamp = datetime.now().isoformat()

        if format == "markdown":
            report = self._format_markdown_report(analysis, timestamp)
        elif format == "html":
            report = self._format_html_report(analysis, timestamp)
        else:  # json
            report = json.dumps(analysis, indent=2, default=str)

        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report)

        return report

    def _format_markdown_report(self, analysis: dict, timestamp: str) -> str:
        """Format analysis as markdown."""
        lines = [
            "# Data Analysis Report",
            f"\n*Generated: {timestamp}*\n",
        ]

        if "shape" in analysis:
            lines.append(f"## Dataset Overview\n")
            lines.append(f"- **Rows:** {analysis['shape'][0]}")
            lines.append(f"- **Columns:** {analysis['shape'][1]}")
            if analysis.get("sampled"):
                lines.append("- ⚠️ *Data was sampled due to size*")

        if "columns" in analysis:
            lines.append("\n## Columns\n")
            for col in analysis["columns"]:
                dtype = analysis.get("dtypes", {}).get(col, "unknown")
                lines.append(f"- `{col}` ({dtype})")

        if "null_counts" in analysis:
            null_cols = [k for k, v in analysis["null_counts"].items() if v > 0]
            if null_cols:
                lines.append("\n## Missing Values\n")
                for col in null_cols:
                    lines.append(f"- `{col}`: {analysis['null_counts'][col]} nulls")

        if "summary" in analysis:
            lines.append("\n## Summary Statistics\n")
            lines.append("```")
            # Simplified summary
            for col, stats in list(analysis["summary"].items())[:5]:
                if isinstance(stats, dict):
                    count = stats.get("count", "N/A")
                    mean = stats.get("mean", "N/A")
                    if mean != "N/A" and mean is not None:
                        lines.append(f"{col}: count={count}, mean={mean:.2f}")
                    else:
                        lines.append(f"{col}: count={count}")
            lines.append("```")

        return "\n".join(lines)

    def _format_html_report(self, analysis: dict, timestamp: str) -> str:
        """Format analysis as HTML."""
        pd = _get_pandas()

        html = [
            "<!DOCTYPE html>",
            "<html><head><style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "</style></head><body>",
            "<h1>Data Analysis Report</h1>",
            f"<p><em>Generated: {timestamp}</em></p>",
        ]

        if "shape" in analysis:
            html.append(f"<p><strong>Shape:</strong> {analysis['shape']}</p>")

        if "sample_rows" in analysis:
            df = pd.DataFrame(analysis["sample_rows"])
            html.append("<h2>Sample Data</h2>")
            html.append(df.to_html(index=False))

        html.append("</body></html>")
        return "\n".join(html)

    async def compare_files(
        self,
        file1: str,
        file2: str,
    ) -> dict[str, Any]:
        """
        Compare two data files.

        Args:
            file1: Path to first file.
            file2: Path to second file.

        Returns:
            Comparison results.
        """
        pd = _get_pandas()

        loop = asyncio.get_event_loop()

        def _compare():
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)

            return {
                "file1_shape": df1.shape,
                "file2_shape": df2.shape,
                "common_columns": list(set(df1.columns) & set(df2.columns)),
                "unique_to_file1": list(set(df1.columns) - set(df2.columns)),
                "unique_to_file2": list(set(df2.columns) - set(df1.columns)),
            }

        return await loop.run_in_executor(None, _compare)


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test():
        import tempfile

        analyzer = DataAnalyzerSkill()

        # Create test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,score\n")
            f.write("Alice,30,85\n")
            f.write("Bob,25,90\n")
            f.write("Charlie,35,78\n")
            test_file = f.name

        # Analyze
        stats = await analyzer.analyze_csv(test_file)
        print(f"✓ Analyzed CSV: shape={stats['shape']}")

        # Create report
        report = await analyzer.create_report(stats)
        print(f"✓ Created report: {len(report)} chars")

        # Generate chart
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            chart_path = f.name

        chart = await analyzer.generate_chart(
            test_file,
            chart_type="bar",
            output=chart_path,
            x_column="name",
            y_column="score",
        )
        print(f"✓ Generated chart: {chart}")

        print("\n✓ All tests passed!")

    asyncio.run(test())
