"""
OpenRouter API client for Vision LLM interactions.

Provides async interface to OpenRouter's chat completions API with
support for vision models and JSON response format.
"""

import asyncio
import base64
import json
import logging
import re
from typing import Any, Optional

import httpx
from pydantic import ValidationError

from ..utils.config import settings
from .json_repair import (
    repair_json,
    extract_json_from_response,
    save_debug_json,
)

logger = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""

    pass


class OpenRouterRateLimitError(OpenRouterError):
    """Raised when rate limited by the API."""

    pass


class OpenRouterClient:
    """
    Async client for OpenRouter API with vision support.

    Features:
    - Vision model support (image inputs)
    - JSON response format enforcement
    - Automatic retry with exponential backoff
    - Prompt caching support (via consistent system prompts)
    - Multi-level fallback support (primary → secondary → tertiary)
    - JSON repair for malformed responses
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        secondary_fallback_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to settings)
            model: Model identifier (defaults to settings.vlm_model)
            fallback_model: First fallback model for when primary fails
            secondary_fallback_model: Second fallback model (e.g., text model)
            timeout: Request timeout in seconds (defaults to settings.vlm_timeout)
        """
        from ..utils.config import get_openrouter_api_key

        self.api_key = api_key or get_openrouter_api_key()
        self.model = model or settings.vlm_model
        self.fallback_model = fallback_model or settings.vlm_fallback_model
        self.secondary_fallback_model = secondary_fallback_model
        self.timeout = timeout or settings.vlm_timeout

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pdf-extractor",
            "X-Title": "PDF Extractor",
        }

    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 data URL."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[bytes] | None = None,
    ) -> list[dict]:
        """Build the messages payload for the API request."""
        if images:
            # Build image content blocks
            image_contents = [
                {"type": "image_url", "image_url": {"url": self._encode_image(img)}}
                for img in images
            ]
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        *image_contents,
                    ],
                },
            ]
        else:
            # Text-only messages
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

    async def _make_request(
        self,
        messages: list[dict],
        temperature: float,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Make a single API request."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            # Note: response_format removed - causes 404 with some OpenRouter data policies
        }

        # Add max_tokens if specified (important for long extractions)
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.BASE_URL,
                headers=self.headers,
                json=payload,
            )

            if response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limited by OpenRouter API")

            response.raise_for_status()
            return response.json()

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """
        Parse JSON from response content with repair strategies.

        Args:
            content: Raw response content

        Returns:
            Parsed JSON dict

        Raises:
            json.JSONDecodeError: If parsing fails even after repair
        """
        # Extract and clean JSON
        cleaned = extract_json_from_response(content)

        # Try direct parse first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try repair and parse
        repaired = repair_json(cleaned)
        try:
            result = json.loads(repaired)
            logger.info("JSON parsed successfully after repair")
            print("🔧 JSON repaired successfully")
            return result
        except json.JSONDecodeError:
            pass

        # Last resort: try to extract just the valid part
        # Find the largest valid JSON substring
        for end_pos in range(len(repaired), 100, -100):
            try:
                truncated = repaired[:end_pos]
                # Try to close it properly
                open_braces = truncated.count('{') - truncated.count('}')
                open_brackets = truncated.count('[') - truncated.count(']')
                truncated = re.sub(r',\s*$', '', truncated.rstrip())
                truncated += ']' * open_brackets + '}' * open_braces
                result = json.loads(truncated)
                logger.warning(f"JSON parsed by truncating to {end_pos} chars")
                print(f"🔧 JSON parsed by truncating (lost some data)")
                return result
            except (json.JSONDecodeError, Exception):
                continue

        # Re-raise original error with cleaned content
        return json.loads(cleaned)

    async def _try_model_extraction(
        self,
        messages: list[dict],
        temperature: float,
        model: str,
        model_name: str = "model",
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[dict[str, Any]], Optional[Exception], Optional[str]]:
        """
        Try extraction with a specific model.

        Returns:
            Tuple of (result, error, raw_content)
        """
        try:
            result = await self._make_request(messages, temperature, model=model, max_tokens=max_tokens)

            choices = result.get("choices", [])
            if not choices:
                raise ValueError(f"No choices in response: {result}")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError(f"Empty content in response: {result}")

            parsed = self._parse_json_response(content)
            return parsed, None, content

        except json.JSONDecodeError as e:
            return None, e, content if 'content' in dir() else None
        except Exception as e:
            return None, e, None

    async def extract_with_vision(
        self,
        images: list[bytes],
        system_prompt: str,
        user_prompt: str,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Send images to the VLM and extract structured JSON data.

        Args:
            images: List of PNG images as bytes
            system_prompt: System instructions (cached by OpenRouter)
            user_prompt: User prompt describing the extraction task
            max_retries: Number of retry attempts (defaults to settings)
            temperature: Model temperature (defaults to settings)
            max_tokens: Maximum tokens for response (defaults to settings.vlm_max_tokens)

        Returns:
            Parsed JSON response from the model

        Raises:
            OpenRouterError: If all retries fail
            json.JSONDecodeError: If response is not valid JSON
        """
        max_retries = max_retries or settings.vlm_max_retries
        temperature = temperature if temperature is not None else settings.vlm_temperature
        max_tokens = max_tokens or settings.vlm_max_tokens

        messages = self._build_messages(system_prompt, user_prompt, images)
        last_error: Optional[Exception] = None
        last_content: Optional[str] = None

        # Try primary model with retries
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"VLM request attempt {attempt + 1}/{max_retries} "
                    f"(model={self.model}, temp={temperature:.2f}, max_tokens={max_tokens})"
                )

                result = await self._make_request(messages, temperature, max_tokens=max_tokens)

                choices = result.get("choices", [])
                if not choices:
                    raise ValueError(f"No choices in response: {result}")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty content in response: {result}")

                last_content = content
                parsed = self._parse_json_response(content)

                logger.info("VLM extraction successful")
                return parsed

            except OpenRouterRateLimitError:
                wait_time = 30 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)
                last_error = OpenRouterRateLimitError("Rate limited")

            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                last_error = e
                await asyncio.sleep(2**attempt)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                last_error = e

                error_details = f"Line {e.lineno}, column {e.colno}: {e.msg}"
                debug_path = save_debug_json(
                    last_content or "[No content]",
                    "json_parse",
                    error_details,
                )

                print("\n" + "=" * 70)
                print(f"📋 Parse error at line {e.lineno}, column {e.colno}:")
                print(f"   {e.msg}")
                print(f"💾 Saved to: {debug_path}")
                print("=" * 70 + "\n")

                temperature = min(temperature + 0.1, 1.0)
                await asyncio.sleep(2**attempt)

            except Exception as e:
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_error = e
                await asyncio.sleep(2**attempt)

        # Try fallback model
        if self.fallback_model and self.fallback_model != self.model:
            print("\n" + "=" * 70)
            print(f"🔄 PRIMARY MODEL FAILED - Trying fallback: {self.fallback_model}")
            print("=" * 70 + "\n")
            logger.info(f"Attempting fallback model: {self.fallback_model}")

            result, error, content = await self._try_model_extraction(
                messages, temperature, self.fallback_model, "fallback", max_tokens=max_tokens
            )

            if result is not None:
                print("✅ Fallback model succeeded!")
                logger.info("Fallback model extraction successful")
                return result

            logger.warning(f"Fallback model also failed: {error}")
            print(f"❌ Fallback model failed: {error}")

            if isinstance(error, json.JSONDecodeError) and content:
                debug_path = save_debug_json(
                    content, "json_parse_fallback",
                    f"Line {error.lineno}, column {error.colno}: {error.msg}"
                )
                print(f"💾 Saved to: {debug_path}")

            last_error = error

        # Try secondary fallback model (e.g., text model)
        if self.secondary_fallback_model and self.secondary_fallback_model != self.model:
            print("\n" + "=" * 70)
            print(f"🔄 FALLBACK FAILED - Trying secondary: {self.secondary_fallback_model}")
            print("=" * 70 + "\n")
            logger.info(f"Attempting secondary fallback: {self.secondary_fallback_model}")

            # For text models, rebuild messages without images
            text_messages = self._build_messages(system_prompt, user_prompt, images=None)

            result, error, content = await self._try_model_extraction(
                text_messages, temperature, self.secondary_fallback_model, "secondary"
            )

            if result is not None:
                print("✅ Secondary fallback model succeeded!")
                logger.info("Secondary fallback extraction successful")
                return result

            logger.warning(f"Secondary fallback also failed: {error}")
            print(f"❌ Secondary fallback failed: {error}")
            last_error = error

        raise OpenRouterError(
            f"Failed after {max_retries} attempts + fallbacks. Last error: {last_error}"
        )

    async def validate_and_extract(
        self,
        images: list[bytes],
        system_prompt: str,
        user_prompt: str,
        model_class: type,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Extract data and validate against a Pydantic model.

        Similar to extract_with_vision but includes Pydantic validation.
        Will retry on validation errors.

        Args:
            images: List of PNG images as bytes
            system_prompt: System instructions
            user_prompt: User prompt
            model_class: Pydantic model class for validation
            max_retries: Number of retry attempts
            temperature: Model temperature
            max_tokens: Maximum tokens for response (defaults to settings.vlm_max_tokens)

        Returns:
            Validated Pydantic model instance

        Raises:
            OpenRouterError: If all retries fail
            ValidationError: If response doesn't match the model
        """
        max_retries = max_retries or settings.vlm_max_retries
        temperature = temperature if temperature is not None else settings.vlm_temperature
        max_tokens = max_tokens or settings.vlm_max_tokens

        messages = self._build_messages(system_prompt, user_prompt, images)
        last_error: Optional[Exception] = None
        last_content: Optional[str] = None  # Track raw content for debug

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"VLM validated request attempt {attempt + 1}/{max_retries} "
                    f"(max_tokens={max_tokens})"
                )

                result = await self._make_request(messages, temperature, max_tokens=max_tokens)

                # Extract content from response
                choices = result.get("choices", [])
                if not choices:
                    raise ValueError(f"No choices in response: {result}")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty content in response: {result}")

                # Handle potential markdown code blocks
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Save for debug before parsing
                last_content = content

                parsed = json.loads(content)

                # Validate with Pydantic
                validated = model_class(**parsed)
                logger.info("VLM extraction and validation successful")
                return validated

            except ValidationError as e:
                logger.warning(f"Validation error on attempt {attempt + 1}: {e}")
                last_error = e

                # Build error details for saving
                error_lines = []
                for error in e.errors():
                    loc = " → ".join(str(x) for x in error["loc"])
                    error_lines.append(f"{loc}: {error['msg']} ({error['type']})")
                error_details = "\n".join(error_lines)

                # Save debug file with parsed JSON
                try:
                    content_to_save = json.dumps(parsed, indent=2, ensure_ascii=False)
                except Exception:
                    content_to_save = str(parsed)
                debug_path = save_debug_json(content_to_save, "validation", error_details)

                # Debug output: show raw JSON and detailed Pydantic errors
                print("\n" + "=" * 70)
                print("❌ VALIDATION FAILED - Raw JSON from VLM:")
                print("=" * 70)
                try:
                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
                except Exception:
                    print(f"[Could not format JSON: {parsed}]")
                print("=" * 70)
                print("📋 PYDANTIC VALIDATION ERRORS:")
                print("=" * 70)
                for error in e.errors():
                    loc = " → ".join(str(x) for x in error["loc"])
                    print(f"  Field: {loc}")
                    print(f"  Type:  {error['type']}")
                    print(f"  Msg:   {error['msg']}")
                    if "input" in error:
                        input_val = error["input"]
                        if isinstance(input_val, str) and len(input_val) > 100:
                            input_val = input_val[:100] + "..."
                        print(f"  Input: {input_val}")
                    print()
                print(f"💾 Saved to: {debug_path}")
                print("=" * 70 + "\n")

                temperature = min(temperature + 0.15, 1.0)
                await asyncio.sleep(2**attempt)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                last_error = e

                # Save debug file
                error_details = f"Line {e.lineno}, column {e.colno}: {e.msg}"
                debug_path = save_debug_json(
                    last_content or "[No content]",
                    "json_parse",
                    error_details,
                )

                # Debug output: show raw content that failed to parse
                print("\n" + "=" * 70)
                print("❌ JSON PARSE ERROR - Raw content from VLM:")
                print("=" * 70)
                if last_content:
                    print(last_content)
                else:
                    print("[No content captured]")
                print("=" * 70)
                print(f"📋 Parse error at line {e.lineno}, column {e.colno}:")
                print(f"   {e.msg}")
                print(f"💾 Saved to: {debug_path}")
                print("=" * 70 + "\n")

                temperature = min(temperature + 0.1, 1.0)
                await asyncio.sleep(2**attempt)

            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
                last_error = e
                await asyncio.sleep(2**attempt)

        # Try fallback model if primary failed
        if self.fallback_model and self.fallback_model != self.model:
            print("\n" + "=" * 70)
            print(f"🔄 PRIMARY MODEL FAILED - Trying fallback: {self.fallback_model}")
            print("=" * 70 + "\n")
            logger.info(f"Attempting fallback model: {self.fallback_model}")

            try:
                result = await self._make_request(messages, temperature, model=self.fallback_model, max_tokens=max_tokens)

                choices = result.get("choices", [])
                if not choices:
                    raise ValueError(f"No choices in response: {result}")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty content in response: {result}")

                # Handle potential markdown code blocks
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                parsed = json.loads(content)

                # Validate with Pydantic
                validated = model_class(**parsed)

                print("✅ Fallback model succeeded!")
                logger.info("Fallback model extraction and validation successful")
                return validated

            except ValidationError as fallback_error:
                logger.warning(f"Fallback model validation failed: {fallback_error}")
                print(f"❌ Fallback model validation failed")

                # Save debug
                try:
                    content_to_save = json.dumps(parsed, indent=2, ensure_ascii=False)
                except Exception:
                    content_to_save = str(parsed) if 'parsed' in dir() else "[No content]"

                error_lines = [f"{' → '.join(str(x) for x in err['loc'])}: {err['msg']}" for err in fallback_error.errors()]
                debug_path = save_debug_json(content_to_save, "validation_fallback", "\n".join(error_lines))
                print(f"💾 Saved to: {debug_path}")

            except Exception as fallback_error:
                logger.warning(f"Fallback model also failed: {fallback_error}")
                print(f"❌ Fallback model also failed: {fallback_error}")

                if isinstance(fallback_error, json.JSONDecodeError):
                    debug_path = save_debug_json(
                        content if 'content' in dir() else "[No content]",
                        "json_parse_fallback",
                        f"Line {fallback_error.lineno}, column {fallback_error.colno}: {fallback_error.msg}",
                    )
                    print(f"💾 Saved to: {debug_path}")

        raise OpenRouterError(
            f"Failed after {max_retries} attempts (+ fallback). Last error: {last_error}"
        )

    async def extract_with_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Send text-only prompt to the LLM and extract structured JSON data.

        Useful for parsing/processing text data without images.

        Args:
            system_prompt: System instructions
            user_prompt: User prompt with data to process
            max_retries: Number of retry attempts (defaults to settings)
            temperature: Model temperature (defaults to settings)
            model: Optional model override (defaults to self.model)
            max_tokens: Maximum tokens for response (defaults to settings.vlm_max_tokens)

        Returns:
            Parsed JSON response from the model

        Raises:
            OpenRouterError: If all retries fail
            json.JSONDecodeError: If response is not valid JSON
        """
        max_retries = max_retries or settings.vlm_max_retries
        temperature = temperature if temperature is not None else settings.vlm_temperature
        max_tokens = max_tokens or settings.vlm_max_tokens
        original_model = self.model

        if model:
            self.model = model

        try:
            messages = self._build_messages(system_prompt, user_prompt, images=None)
            last_error: Optional[Exception] = None
            last_content: Optional[str] = None  # Track raw content for debug

            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Text LLM request attempt {attempt + 1}/{max_retries} "
                        f"(temp={temperature:.2f}, max_tokens={max_tokens})"
                    )

                    result = await self._make_request(messages, temperature, max_tokens=max_tokens)

                    # Extract content from response
                    choices = result.get("choices", [])
                    if not choices:
                        raise ValueError(f"No choices in response: {result}")

                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        raise ValueError(f"Empty content in response: {result}")

                    # Handle potential markdown code blocks
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    # Save for debug before parsing
                    last_content = content

                    parsed = json.loads(content)

                    logger.info("Text LLM extraction successful")
                    return parsed

                except OpenRouterRateLimitError:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                    last_error = OpenRouterRateLimitError("Rate limited")

                except httpx.HTTPStatusError as e:
                    logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                    last_error = e
                    await asyncio.sleep(2**attempt)

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                    last_error = e

                    # Save debug file
                    error_details = f"Line {e.lineno}, column {e.colno}: {e.msg}"
                    debug_path = save_debug_json(
                        last_content or "[No content]",
                        "json_parse",
                        error_details,
                    )

                    # Debug output: show raw content that failed to parse
                    print("\n" + "=" * 70)
                    print("❌ JSON PARSE ERROR - Raw content from LLM:")
                    print("=" * 70)
                    if last_content:
                        print(last_content)
                    else:
                        print("[No content captured]")
                    print("=" * 70)
                    print(f"📋 Parse error at line {e.lineno}, column {e.colno}:")
                    print(f"   {e.msg}")
                    print(f"💾 Saved to: {debug_path}")
                    print("=" * 70 + "\n")

                    temperature = min(temperature + 0.1, 1.0)
                    await asyncio.sleep(2**attempt)

                except Exception as e:
                    logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                    last_error = e
                    await asyncio.sleep(2**attempt)

            raise OpenRouterError(
                f"Failed after {max_retries} attempts. Last error: {last_error}"
            )
        finally:
            self.model = original_model
