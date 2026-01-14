"""
Unit tests for BrowserAutomationSkill.

Uses pytest and pytest-asyncio for async test support.
Tests use mocking to avoid actual browser operations in CI.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


# Import the skill
from skills.browser_automation import BrowserAutomationSkill


class TestBrowserAutomationSkillInit:
    """Tests for BrowserAutomationSkill initialization."""

    def test_default_init(self):
        """Test default initialization parameters."""
        skill = BrowserAutomationSkill()
        assert skill.headless is True
        assert skill.timeout == 30000
        assert skill.user_agent is None

    def test_custom_init(self):
        """Test custom initialization parameters."""
        skill = BrowserAutomationSkill(
            headless=False,
            timeout=10000,
            user_agent="CustomAgent/1.0",
        )
        assert skill.headless is False
        assert skill.timeout == 10000
        assert skill.user_agent == "CustomAgent/1.0"

    def test_env_override_headless_false(self, monkeypatch):
        """Test BROWSER_HEADLESS=false env var override."""
        monkeypatch.setenv("BROWSER_HEADLESS", "false")
        skill = BrowserAutomationSkill(headless=True)
        assert skill.headless is False

    def test_env_override_headless_true(self, monkeypatch):
        """Test BROWSER_HEADLESS=true env var override."""
        monkeypatch.setenv("BROWSER_HEADLESS", "true")
        skill = BrowserAutomationSkill(headless=False)
        assert skill.headless is True


class TestNavigate:
    """Tests for navigate method."""

    @pytest.mark.asyncio
    async def test_navigate_empty_url_raises(self):
        """Test that empty URL raises ValueError."""
        skill = BrowserAutomationSkill()
        with pytest.raises(ValueError, match="URL cannot be empty"):
            await skill.navigate("")

    @pytest.mark.asyncio
    async def test_navigate_success(self):
        """Test successful navigation with mocked browser."""
        skill = BrowserAutomationSkill()

        # Mock the page object
        mock_page = AsyncMock()
        mock_page.url = "https://example.com"
        mock_page.title = AsyncMock(return_value="Example Domain")
        mock_response = MagicMock()
        mock_response.status = 200
        mock_page.goto = AsyncMock(return_value=mock_response)

        skill._page = mock_page

        result = await skill.navigate("https://example.com")

        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example Domain"
        assert result["status_code"] == 200


class TestClick:
    """Tests for click method."""

    @pytest.mark.asyncio
    async def test_click_empty_selector_raises(self):
        """Test that empty selector raises ValueError."""
        skill = BrowserAutomationSkill()
        with pytest.raises(ValueError, match="Selector cannot be empty"):
            await skill.click("")

    @pytest.mark.asyncio
    async def test_click_success(self):
        """Test successful click with mocked page."""
        skill = BrowserAutomationSkill()

        mock_page = AsyncMock()
        mock_page.click = AsyncMock()
        skill._page = mock_page

        result = await skill.click("button.submit")

        assert result["success"] is True
        assert result["selector"] == "button.submit"
        mock_page.click.assert_called_once()


class TestFillForm:
    """Tests for fill_form method."""

    @pytest.mark.asyncio
    async def test_fill_form_empty_data_raises(self):
        """Test that empty form data raises ValueError."""
        skill = BrowserAutomationSkill()
        with pytest.raises(ValueError, match="Form data cannot be empty"):
            await skill.fill_form({})

    @pytest.mark.asyncio
    async def test_fill_form_success(self):
        """Test successful form filling."""
        skill = BrowserAutomationSkill()

        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()
        skill._page = mock_page

        form_data = {
            "#username": "testuser",
            "#password": "secret123",
        }
        result = await skill.fill_form(form_data)

        assert result["success"] is True
        assert "#username" in result["filled_fields"]
        assert "#password" in result["filled_fields"]
        assert len(result["failed_fields"]) == 0

    @pytest.mark.asyncio
    async def test_fill_form_with_submit(self):
        """Test form filling with submit button."""
        skill = BrowserAutomationSkill()

        mock_page = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.click = AsyncMock()
        skill._page = mock_page

        result = await skill.fill_form(
            {"#email": "test@example.com"},
            submit_selector="button[type='submit']",
        )

        assert result["success"] is True
        assert "submit" in result
        mock_page.click.assert_called_once()


class TestExtractContent:
    """Tests for extract_content method."""

    @pytest.mark.asyncio
    async def test_extract_empty_selector_raises(self):
        """Test that empty selector raises ValueError."""
        skill = BrowserAutomationSkill()
        with pytest.raises(ValueError, match="Selector cannot be empty"):
            await skill.extract_content("")

    @pytest.mark.asyncio
    async def test_extract_text_success(self):
        """Test successful text extraction."""
        skill = BrowserAutomationSkill()

        mock_element = AsyncMock()
        mock_element.text_content = AsyncMock(return_value="Hello World")

        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(return_value=mock_element)
        skill._page = mock_page

        result = await skill.extract_content("h1")

        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_extract_attribute_success(self):
        """Test successful attribute extraction."""
        skill = BrowserAutomationSkill()

        mock_element = AsyncMock()
        mock_element.get_attribute = AsyncMock(return_value="https://example.com")

        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(return_value=mock_element)
        skill._page = mock_page

        result = await skill.extract_content("a.link", attribute="href")

        assert result == "https://example.com"


class TestScreenshot:
    """Tests for screenshot method."""

    @pytest.mark.asyncio
    async def test_screenshot_empty_path_raises(self):
        """Test that empty path raises ValueError."""
        skill = BrowserAutomationSkill()
        with pytest.raises(ValueError, match="Screenshot path cannot be empty"):
            await skill.screenshot("")

    @pytest.mark.asyncio
    async def test_screenshot_full_page(self, tmp_path):
        """Test full page screenshot."""
        skill = BrowserAutomationSkill()

        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()
        skill._page = mock_page

        screenshot_path = str(tmp_path / "test.png")
        result = await skill.screenshot(screenshot_path, full_page=True)

        assert Path(result).name == "test.png"
        mock_page.screenshot.assert_called_once()


class TestWaitForElement:
    """Tests for wait_for_element method."""

    @pytest.mark.asyncio
    async def test_wait_empty_selector_raises(self):
        """Test that empty selector raises ValueError."""
        skill = BrowserAutomationSkill()
        with pytest.raises(ValueError, match="Selector cannot be empty"):
            await skill.wait_for_element("")

    @pytest.mark.asyncio
    async def test_wait_success(self):
        """Test successful wait for element."""
        skill = BrowserAutomationSkill()

        mock_page = AsyncMock()
        mock_page.wait_for_selector = AsyncMock(return_value=MagicMock())
        skill._page = mock_page

        result = await skill.wait_for_element(".loaded")

        assert result is True


class TestGetPageText:
    """Tests for get_page_text method."""

    @pytest.mark.asyncio
    async def test_get_page_text_success(self):
        """Test successful page text extraction."""
        skill = BrowserAutomationSkill()

        mock_page = AsyncMock()
        mock_page.inner_text = AsyncMock(return_value="Page content here")
        skill._page = mock_page

        result = await skill.get_page_text()

        assert result == "Page content here"


class TestCookies:
    """Tests for cookie management methods."""

    @pytest.mark.asyncio
    async def test_get_cookies(self):
        """Test getting cookies."""
        skill = BrowserAutomationSkill()

        mock_context = AsyncMock()
        mock_context.cookies = AsyncMock(
            return_value=[{"name": "session", "value": "abc123"}]
        )
        skill._context = mock_context

        cookies = await skill.get_cookies()

        assert len(cookies) == 1
        assert cookies[0]["name"] == "session"

    @pytest.mark.asyncio
    async def test_set_cookies(self):
        """Test setting cookies."""
        skill = BrowserAutomationSkill()

        mock_context = AsyncMock()
        mock_context.add_cookies = AsyncMock()
        skill._context = mock_context

        await skill.set_cookies(
            [{"name": "test", "value": "value", "url": "https://example.com"}]
        )

        mock_context.add_cookies.assert_called_once()


class TestCloseBrowser:
    """Tests for close_browser method."""

    @pytest.mark.asyncio
    async def test_close_browser_cleanup(self):
        """Test that close_browser cleans up all resources."""
        skill = BrowserAutomationSkill()

        mock_page = AsyncMock()
        mock_context = AsyncMock()
        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()

        skill._page = mock_page
        skill._context = mock_context
        skill._browser = mock_browser
        skill._playwright = mock_playwright

        await skill.close_browser()

        mock_page.close.assert_called_once()
        mock_context.close.assert_called_once()
        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

        assert skill._page is None
        assert skill._context is None
        assert skill._browser is None
        assert skill._playwright is None


class TestAsyncContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test that context manager properly cleans up."""
        skill = BrowserAutomationSkill()

        # Mock _ensure_browser and close_browser
        skill._ensure_browser = AsyncMock()
        skill.close_browser = AsyncMock()

        async with skill:
            pass

        skill._ensure_browser.assert_called_once()
        skill.close_browser.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST (requires actual browser - skip in CI)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_navigate_example_com():
    """
    Integration test that actually navigates to example.com.

    Run with: pytest -m integration tests/test_browser_automation.py
    """
    async with BrowserAutomationSkill(headless=True) as browser:
        result = await browser.navigate("https://example.com")

        assert result["success"] is True
        assert "Example" in result["title"]

        text = await browser.get_page_text()
        assert len(text) > 0
