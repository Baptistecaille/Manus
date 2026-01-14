"""
Browser Automation Skill - Playwright-based browser automation for Manus agent.

Provides comprehensive browser automation capabilities including navigation,
form interaction, content extraction, and screenshot capture.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
)

# Configure logging based on LOG_LEVEL env var
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)


class BrowserAutomationSkill:
    """
    Skill for browser automation using Playwright.

    Provides async methods for web automation including navigation,
    clicking, form filling, content extraction, and screenshots.

    Attributes:
        headless: Whether to run browser in headless mode.
        timeout: Default timeout for operations in milliseconds.

    Example:
        >>> skill = BrowserAutomationSkill()
        >>> await skill.navigate("https://example.com")
        >>> text = await skill.get_page_text()
        >>> await skill.close_browser()
    """

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Initialize the browser automation skill.

        Args:
            headless: Run browser in headless mode. Default True.
                     Can be overridden by BROWSER_HEADLESS env var.
            timeout: Default timeout in ms for operations. Default 30000.
            user_agent: Custom user agent string. Default None uses browser default.
        """
        # Allow env var override for headless mode
        env_headless = os.getenv("BROWSER_HEADLESS", "").lower()
        if env_headless in ("false", "0", "no"):
            self.headless = False
        elif env_headless in ("true", "1", "yes"):
            self.headless = True
        else:
            self.headless = headless

        self.timeout = timeout
        self.user_agent = user_agent

        self._playwright: Any = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        logger.debug(
            f"BrowserAutomationSkill initialized (headless={self.headless}, timeout={self.timeout})"
        )

    async def _ensure_browser(self) -> Page:
        """
        Ensure browser, context, and page are initialized.

        Returns:
            The active Page instance.

        Raises:
            RuntimeError: If browser initialization fails.
        """
        if self._page is not None:
            return self._page

        try:
            logger.debug("Initializing Playwright browser...")
            self._playwright = await async_playwright().start()

            # Use Chromium for best compatibility
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
            )

            # Create context with optional user agent
            context_options: dict[str, Any] = {}
            if self.user_agent:
                context_options["user_agent"] = self.user_agent

            self._context = await self._browser.new_context(**context_options)
            self._context.set_default_timeout(self.timeout)

            self._page = await self._context.new_page()
            logger.info("Browser initialized successfully")
            return self._page

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise RuntimeError(f"Browser initialization failed: {e}") from e

    async def navigate(self, url: str, wait_until: str = "load") -> dict[str, Any]:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to.
            wait_until: When to consider navigation complete.
                       Options: 'load', 'domcontentloaded', 'networkidle', 'commit'.

        Returns:
            Dict with status, url, and title.

        Raises:
            ValueError: If URL is empty.
            RuntimeError: If navigation fails.

        Example:
            >>> result = await skill.navigate("https://example.com")
            >>> print(result["title"])
        """
        if not url:
            raise ValueError("URL cannot be empty")

        page = await self._ensure_browser()
        logger.info(f"Navigating to: {url}")

        try:
            response = await page.goto(url, wait_until=wait_until)

            result = {
                "success": True,
                "url": page.url,
                "title": await page.title(),
                "status_code": response.status if response else None,
            }
            logger.debug(f"Navigation successful: {result}")
            return result

        except PlaywrightTimeoutError as e:
            logger.warning(f"Navigation timeout for {url}: {e}")
            return {
                "success": False,
                "error": "timeout",
                "message": f"Navigation timed out after {self.timeout}ms",
                "url": url,
            }
        except Exception as e:
            logger.error(f"Navigation failed for {url}: {e}")
            raise RuntimeError(f"Navigation failed: {e}") from e

    async def click(
        self,
        selector: str,
        timeout: Optional[int] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Click an element on the page.

        Args:
            selector: CSS selector or XPath for the element.
            timeout: Custom timeout in ms. Uses default if None.
            force: Force click even if element is not visible.

        Returns:
            Dict with success status and details.

        Raises:
            ValueError: If selector is empty.

        Example:
            >>> await skill.click("button.submit")
            >>> await skill.click("//button[@type='submit']")
        """
        if not selector:
            raise ValueError("Selector cannot be empty")

        page = await self._ensure_browser()
        logger.info(f"Clicking element: {selector}")

        try:
            await page.click(
                selector,
                timeout=timeout or self.timeout,
                force=force,
            )
            return {
                "success": True,
                "selector": selector,
                "message": "Click successful",
            }

        except PlaywrightTimeoutError:
            logger.warning(f"Click timeout for selector: {selector}")
            return {
                "success": False,
                "error": "timeout",
                "selector": selector,
                "message": f"Element not found within timeout",
            }
        except Exception as e:
            logger.error(f"Click failed for {selector}: {e}")
            return {
                "success": False,
                "error": "click_failed",
                "selector": selector,
                "message": str(e),
            }

    async def fill_form(
        self,
        form_data: dict[str, str],
        submit_selector: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Fill a form with provided data.

        Args:
            form_data: Dict mapping selectors to values to fill.
            submit_selector: Optional selector for submit button.

        Returns:
            Dict with success status and filled fields.

        Example:
            >>> await skill.fill_form({
            ...     "#username": "user@example.com",
            ...     "#password": "secret123"
            ... }, submit_selector="button[type='submit']")
        """
        if not form_data:
            raise ValueError("Form data cannot be empty")

        page = await self._ensure_browser()
        logger.info(f"Filling form with {len(form_data)} fields")

        filled_fields: list[str] = []
        failed_fields: list[dict[str, str]] = []

        for selector, value in form_data.items():
            try:
                await page.fill(selector, value)
                filled_fields.append(selector)
                logger.debug(f"Filled field: {selector}")
            except Exception as e:
                logger.warning(f"Failed to fill {selector}: {e}")
                failed_fields.append({"selector": selector, "error": str(e)})

        result: dict[str, Any] = {
            "success": len(failed_fields) == 0,
            "filled_fields": filled_fields,
            "failed_fields": failed_fields,
        }

        # Submit if requested
        if submit_selector:
            submit_result = await self.click(submit_selector)
            result["submit"] = submit_result

        return result

    async def extract_content(
        self,
        selector: str,
        attribute: Optional[str] = None,
    ) -> str:
        """
        Extract text or attribute content from an element.

        Args:
            selector: CSS selector for the element.
            attribute: Optional attribute to extract. If None, extracts text content.

        Returns:
            Extracted text or attribute value.

        Raises:
            ValueError: If selector is empty.
            RuntimeError: If element not found.

        Example:
            >>> text = await skill.extract_content("h1.title")
            >>> href = await skill.extract_content("a.link", attribute="href")
        """
        if not selector:
            raise ValueError("Selector cannot be empty")

        page = await self._ensure_browser()
        logger.debug(f"Extracting content from: {selector}")

        try:
            element = await page.wait_for_selector(selector, timeout=self.timeout)
            if element is None:
                raise RuntimeError(f"Element not found: {selector}")

            if attribute:
                content = await element.get_attribute(attribute)
                return content or ""
            else:
                content = await element.text_content()
                return content or ""

        except PlaywrightTimeoutError:
            logger.warning(f"Element not found: {selector}")
            raise RuntimeError(f"Element not found within timeout: {selector}")
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            raise RuntimeError(f"Content extraction failed: {e}") from e

    async def extract_all(
        self,
        selector: str,
        attribute: Optional[str] = None,
    ) -> list[str]:
        """
        Extract content from all matching elements.

        Args:
            selector: CSS selector for the elements.
            attribute: Optional attribute to extract.

        Returns:
            List of extracted text/attribute values.

        Example:
            >>> links = await skill.extract_all("a.item", attribute="href")
            >>> titles = await skill.extract_all("h2.title")
        """
        if not selector:
            raise ValueError("Selector cannot be empty")

        page = await self._ensure_browser()
        logger.debug(f"Extracting all from: {selector}")

        elements = await page.query_selector_all(selector)
        results: list[str] = []

        for element in elements:
            if attribute:
                value = await element.get_attribute(attribute)
            else:
                value = await element.text_content()
            if value:
                results.append(value)

        logger.debug(f"Extracted {len(results)} items")
        return results

    async def screenshot(
        self,
        path: str,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> str:
        """
        Capture a screenshot of the page or element.

        Args:
            path: File path to save the screenshot.
            full_page: Capture the full scrollable page.
            selector: Optional selector to screenshot specific element.

        Returns:
            Absolute path to saved screenshot.

        Raises:
            ValueError: If path is empty.

        Example:
            >>> path = await skill.screenshot("/tmp/page.png", full_page=True)
            >>> element_path = await skill.screenshot("/tmp/button.png", selector=".hero")
        """
        if not path:
            raise ValueError("Screenshot path cannot be empty")

        page = await self._ensure_browser()
        abs_path = str(Path(path).resolve())
        logger.info(f"Taking screenshot: {abs_path}")

        try:
            if selector:
                element = await page.wait_for_selector(selector)
                if element:
                    await element.screenshot(path=abs_path)
                else:
                    raise RuntimeError(f"Element not found: {selector}")
            else:
                await page.screenshot(path=abs_path, full_page=full_page)

            logger.info(f"Screenshot saved: {abs_path}")
            return abs_path

        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            raise RuntimeError(f"Screenshot failed: {e}") from e

    async def wait_for_element(
        self,
        selector: str,
        timeout: Optional[int] = None,
        state: str = "visible",
    ) -> bool:
        """
        Wait for an element to be in specified state.

        Args:
            selector: CSS selector for the element.
            timeout: Custom timeout in ms.
            state: State to wait for: 'attached', 'detached', 'visible', 'hidden'.

        Returns:
            True if element reached state, False if timeout.

        Example:
            >>> if await skill.wait_for_element(".loading", state="hidden"):
            ...     print("Loading complete")
        """
        if not selector:
            raise ValueError("Selector cannot be empty")

        page = await self._ensure_browser()
        logger.debug(f"Waiting for element: {selector} (state={state})")

        try:
            await page.wait_for_selector(
                selector,
                timeout=timeout or self.timeout,
                state=state,  # type: ignore
            )
            return True

        except PlaywrightTimeoutError:
            logger.debug(f"Timeout waiting for {selector}")
            return False

    async def get_page_text(self) -> str:
        """
        Get all visible text content from the page.

        Returns:
            Full text content of the page body.

        Example:
            >>> text = await skill.get_page_text()
            >>> print(f"Page has {len(text)} characters")
        """
        page = await self._ensure_browser()
        logger.debug("Extracting page text")

        try:
            text = await page.inner_text("body")
            logger.debug(f"Extracted {len(text)} characters")
            return text
        except Exception as e:
            logger.warning(f"Failed to get page text: {e}")
            return ""

    async def get_page_html(self) -> str:
        """
        Get the full HTML content of the page.

        Returns:
            Full HTML of the page.
        """
        page = await self._ensure_browser()
        return await page.content()

    async def evaluate_js(self, script: str) -> Any:
        """
        Execute JavaScript on the page and return result.

        Args:
            script: JavaScript code to execute.

        Returns:
            Result of JavaScript execution.

        Example:
            >>> title = await skill.evaluate_js("document.title")
        """
        page = await self._ensure_browser()
        logger.debug(f"Evaluating JS: {script[:100]}...")
        return await page.evaluate(script)

    async def get_cookies(self) -> list[dict[str, Any]]:
        """
        Get all cookies from the current context.

        Returns:
            List of cookie dictionaries.
        """
        if self._context is None:
            await self._ensure_browser()
        assert self._context is not None

        cookies = await self._context.cookies()
        logger.debug(f"Retrieved {len(cookies)} cookies")
        return cookies

    async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
        """
        Set cookies in the current context.

        Args:
            cookies: List of cookie dictionaries with name, value, url/domain.

        Example:
            >>> await skill.set_cookies([
            ...     {"name": "session", "value": "abc123", "url": "https://example.com"}
            ... ])
        """
        if self._context is None:
            await self._ensure_browser()
        assert self._context is not None

        await self._context.add_cookies(cookies)
        logger.debug(f"Set {len(cookies)} cookies")

    async def clear_cookies(self) -> None:
        """Clear all cookies from the current context."""
        if self._context is not None:
            await self._context.clear_cookies()
            logger.debug("Cookies cleared")

    async def go_back(self) -> dict[str, Any]:
        """Navigate back in browser history."""
        page = await self._ensure_browser()
        await page.go_back()
        return {
            "success": True,
            "url": page.url,
            "title": await page.title(),
        }

    async def go_forward(self) -> dict[str, Any]:
        """Navigate forward in browser history."""
        page = await self._ensure_browser()
        await page.go_forward()
        return {
            "success": True,
            "url": page.url,
            "title": await page.title(),
        }

    async def reload(self) -> dict[str, Any]:
        """Reload the current page."""
        page = await self._ensure_browser()
        await page.reload()
        return {
            "success": True,
            "url": page.url,
            "title": await page.title(),
        }

    async def close_browser(self) -> None:
        """
        Close the browser and cleanup resources.

        Should be called when done with browser automation.
        """
        logger.info("Closing browser...")

        try:
            if self._page:
                await self._page.close()
                self._page = None

            if self._context:
                await self._context.close()
                self._context = None

            if self._browser:
                await self._browser.close()
                self._browser = None

            if self._playwright:
                await self._playwright.stop()
                self._playwright = None

            logger.info("Browser closed successfully")

        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")

    async def __aenter__(self) -> "BrowserAutomationSkill":
        """Async context manager entry."""
        await self._ensure_browser()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close_browser()


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    async def test_browser_automation():
        """Quick test of browser automation capabilities."""
        print("=== Browser Automation Skill Test ===\n")

        async with BrowserAutomationSkill(headless=True) as browser:
            # Test navigation
            print("1. Testing navigation...")
            result = await browser.navigate("https://example.com")
            print(f"   Title: {result.get('title')}")
            print(f"   Status: {result.get('status_code')}")

            # Test text extraction
            print("\n2. Testing text extraction...")
            text = await browser.get_page_text()
            print(f"   Page text length: {len(text)} chars")

            # Test element extraction
            print("\n3. Testing element extraction...")
            try:
                h1_text = await browser.extract_content("h1")
                print(f"   H1 content: {h1_text}")
            except Exception as e:
                print(f"   H1 extraction failed: {e}")

            # Test screenshot
            print("\n4. Testing screenshot...")
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    path = await browser.screenshot(f.name)
                    print(f"   Screenshot saved: {path}")
            except Exception as e:
                print(f"   Screenshot failed: {e}")

        print("\n=== Test Complete ===")

    asyncio.run(test_browser_automation())
