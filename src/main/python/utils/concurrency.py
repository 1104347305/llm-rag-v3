from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class ServiceOverloaded(RuntimeError):
    """Raised when a bounded external-service concurrency slot is unavailable."""

    def __init__(self, service: str) -> None:
        self.service = service
        super().__init__(f"{service} is at concurrency capacity")


class AsyncCapacityLimiter:
    def __init__(self, service: str, limit: int, acquire_timeout: float) -> None:
        self.service = service
        self.limit = max(1, limit)
        self.acquire_timeout = max(0.001, acquire_timeout)
        self._semaphore = asyncio.Semaphore(self.limit)

    async def acquire(self) -> None:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.acquire_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise ServiceOverloaded(self.service) from exc

    def release(self) -> None:
        self._semaphore.release()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()
