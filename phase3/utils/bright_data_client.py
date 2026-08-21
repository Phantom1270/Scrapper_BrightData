"""
Bright Data Web Unlocker API client.
Fetches raw HTML from any URL.
1 credit per successful request. 0 for failures.
"""

import time
import requests
from typing import Optional


class BrightDataClient:
    def __init__(
        self,
        api_token: str,
        zone: str = "web_unlocker1",
        requests_per_minute: int = 300,
        max_retries: int = 2,
        timeout: int = 30,
    ):
        self.api_token = api_token
        self.zone = zone
        self.delay = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self.timeout = timeout

        # Tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.credits_used = 0

    def fetch(self, url: str) -> Optional[str]:
        """
        Fetch a single URL's HTML.

        Returns HTML string on success, None on failure.
        Failed requests are NOT charged by Bright Data.
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "zone": self.zone,
            "url": url,
            "format": "raw",
        }

        for attempt in range(self.max_retries + 1):
            self.total_requests += 1

            try:
                response = requests.post(
                    "https://api.brightdata.com/request",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    self.successful_requests += 1
                    self.credits_used += 1
                    return response.text

                elif response.status_code == 429:
                    # Rate limited
                    wait = (2 ** attempt) * 5
                    time.sleep(wait)
                    continue

                elif response.status_code in (502, 503, 504):
                    # Server error — retry
                    time.sleep(2 ** attempt)
                    continue

                else:
                    # Client error — don't retry
                    self.failed_requests += 1
                    return None

            except requests.Timeout:
                if attempt < self.max_retries:
                    continue
                self.failed_requests += 1
                return None

            except requests.ConnectionError:
                if attempt < self.max_retries:
                    time.sleep(2)
                    continue
                self.failed_requests += 1
                return None

            except Exception:
                self.failed_requests += 1
                return None

        self.failed_requests += 1
        return None

    def fetch_batch(
        self,
        urls: list[str],
        progress_callback=None,
    ) -> dict[str, Optional[str]]:
        """
        Fetch multiple URLs with rate limiting.

        Returns: {url: html_or_none}
        """
        results = {}
        total = len(urls)

        for i, url in enumerate(urls):
            html = self.fetch(url)
            results[url] = html

            if progress_callback:
                progress_callback(i + 1, total, url, html is not None)

            # Rate limiting (don't delay after last request)
            if i < total - 1:
                time.sleep(self.delay)

        return results

    @property
    def stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "credits_used": self.credits_used,
        }
