"""Gemini multimodal integration for Multimodal RAG.

Provides:
- GeminiClient: High-level client for generation, streaming, vision, video frame analysis,
  and embeddings using Google Gemini (via google.genai SDK with automatic REST fallback).
- GeminiChatModel: LangChain-compatible BaseChatModel for pipeline generation and streaming.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import threading
from pathlib import Path
from typing import Any, Iterator

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from PIL import Image
from pydantic import PrivateAttr

log = logging.getLogger(__name__)

GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def get_gemini_api_key(settings=None) -> str | None:
    """Retrieve Gemini API key from settings or environment."""
    if settings and getattr(settings, "gemini_api_key", None):
        return settings.gemini_api_key
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or (getattr(settings, "llm_api_key", None) if settings else None)
    )


def _image_to_part(img_input: str | Path | Image.Image | bytes) -> dict:
    """Convert an image path, bytes, or PIL Image to a Gemini API inlineData dict."""
    mime_type = "image/jpeg"
    if isinstance(img_input, (str, Path)):
        p = Path(img_input)
        mime_type = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        data_bytes = p.read_bytes()
    elif isinstance(img_input, Image.Image):
        buf = io.BytesIO()
        img_input.convert("RGB").save(buf, format="JPEG", quality=85)
        data_bytes = buf.getvalue()
        mime_type = "image/jpeg"
    elif isinstance(img_input, bytes):
        data_bytes = img_input
    else:
        raise ValueError(f"Unsupported image type: {type(img_input)}")

    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(data_bytes).decode("utf-8"),
        }
    }


class GeminiClient:
    """Robust Gemini client wrapping the official google.genai SDK with REST fallback."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or get_gemini_api_key()
        self.model = model
        self._sdk_client = None
        self._init_sdk()

    def _init_sdk(self):
        if not self.api_key:
            return
        try:
            from google import genai

            self._sdk_client = genai.Client(api_key=self.api_key)
        except Exception as e:
            log.warning("Could not initialize google.genai SDK (%s), using direct REST client", e)
            self._sdk_client = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    # ---- Text & Multimodal Generation --------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        images: list[str | Path | Image.Image] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        # Try google.genai SDK
        if self._sdk_client:
            try:
                from google.genai import types

                contents: list[Any] = []
                if images:
                    for img in images:
                        if isinstance(img, (str, Path)):
                            contents.append(Image.open(img).convert("RGB"))
                        elif isinstance(img, Image.Image):
                            contents.append(img.convert("RGB"))
                contents.append(prompt)

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction,
                )
                res = self._sdk_client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                if res and res.text:
                    return res.text.strip()
            except Exception as e:
                log.warning("google.genai SDK generate failed (%s), falling back to REST", e)

        # REST fallback
        return self._generate_rest(prompt, system_instruction, images, temperature, max_tokens)

    def _generate_rest(
        self,
        prompt: str,
        system_instruction: str | None = None,
        images: list[str | Path | Image.Image] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        url = f"{GEMINI_REST_BASE}/{self.model}:generateContent?key={self.api_key}"
        parts: list[dict] = []
        if images:
            for img in images:
                parts.append(_image_to_part(img))
        parts.append({"text": prompt})

        body: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        cand = candidates[0]
        content_parts = cand.get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in content_parts).strip()

    # ---- Streaming ---------------------------------------------------------------------
    def stream(
        self,
        prompt: str,
        system_instruction: str | None = None,
        images: list[str | Path | Image.Image] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        if self._sdk_client:
            try:
                from google.genai import types

                contents: list[Any] = []
                if images:
                    for img in images:
                        if isinstance(img, (str, Path)):
                            contents.append(Image.open(img).convert("RGB"))
                        elif isinstance(img, Image.Image):
                            contents.append(img.convert("RGB"))
                contents.append(prompt)

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction,
                )
                for chunk in self._sdk_client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                ):
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                log.warning("google.genai SDK stream failed (%s), falling back to REST stream", e)

        # REST stream fallback
        url = f"{GEMINI_REST_BASE}/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"
        parts: list[dict] = []
        if images:
            for img in images:
                parts.append(_image_to_part(img))
        parts.append({"text": prompt})

        body: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        with requests.post(url, json=body, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        for cand in data.get("candidates", []):
                            for p in cand.get("content", {}).get("parts", []):
                                t = p.get("text", "")
                                if t:
                                    yield t
                    except Exception:
                        pass

    # ---- Image Description (Photos, Figures, Charts, Diagrams) -------------------------
    def describe_image(self, image_path: str | Path, prompt: str | None = None) -> str:
        prompt = prompt or (
            "Provide a comprehensive, factual description of this image for a multimodal search and retrieval index.\n"
            "Include:\n"
            "1. Overview: What kind of image/photo/document is it and what is depicted?\n"
            "2. Details: Key entities, objects, people, visual elements, scenes, or actions.\n"
            "3. Text & Data: Transcribe any visible text, titles, numbers, labels, charts, tables, or legends.\n"
            "4. Main takeaway: Key insight or purpose of the image.\n"
            "Keep it concise, accurate, and search-friendly (max 200 words)."
        )
        return self.generate(prompt=prompt, images=[image_path], temperature=0.1, max_tokens=400)

    # ---- Video Scene Analysis ----------------------------------------------------------
    def describe_video_scene(
        self,
        frame_paths: list[str | Path],
        timestamp_label: str = "",
        prompt: str | None = None,
    ) -> str:
        prompt = prompt or (
            f"These keyframe images represent a video segment ({timestamp_label}).\n"
            "Describe the scene and visual action clearly for a search index:\n"
            "- What is happening in this segment?\n"
            "- People, objects, environment, actions, motion, or interactions.\n"
            "- Any on-screen text, titles, signs, captions, or graphics visible.\n"
            "Be factual and specific (max 150 words)."
        )
        return self.generate(prompt=prompt, images=frame_paths, temperature=0.1, max_tokens=350)

    # ---- Text Embeddings ---------------------------------------------------------------
    def embed_texts(self, texts: list[str], model: str = "text-embedding-004") -> list[list[float]]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        if self._sdk_client:
            try:
                res = self._sdk_client.models.embed_content(
                    model=model,
                    contents=texts,
                )
                if hasattr(res, "embeddings") and res.embeddings:
                    return [e.values for e in res.embeddings]
            except Exception as e:
                log.warning("google.genai SDK embed failed (%s), falling back to REST", e)

        # REST batch embed fallback
        url = f"{GEMINI_REST_BASE}/{model}:batchEmbedContents?key={self.api_key}"
        reqs = [{"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in texts]
        resp = requests.post(url, json={"requests": reqs}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return [e.get("values", []) for e in data.get("embeddings", [])]


class GeminiChatModel(BaseChatModel):
    """LangChain BaseChatModel implementation powered by Google Gemini."""

    model_name: str = "gemini-2.5-flash"
    api_key: str | None = None
    temperature: float = 0.1
    max_output_tokens: int = 1024

    _client: GeminiClient = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._client = GeminiClient(api_key=self.api_key, model=self.model_name)

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    def _extract_inputs(self, messages: list[BaseMessage]) -> tuple[str, str | None, list[Any]]:
        system_instruction = None
        images: list[Any] = []
        prompt_parts: list[str] = []

        for m in messages:
            if m.type == "system":
                system_instruction = str(m.content)
            elif m.type in ("human", "user"):
                c = m.content
                if isinstance(c, str):
                    prompt_parts.append(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                prompt_parts.append(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                url = part.get("image_url", {}).get("url", "")
                                if url.startswith("data:image"):
                                    # Base64 data URI
                                    header, b64_data = url.split(",", 1)
                                    images.append(base64.b64decode(b64_data))
                            elif part.get("type") == "image_path":
                                p = part.get("path")
                                if p and Path(p).exists():
                                    images.append(Path(p))
            elif m.type in ("ai", "assistant"):
                prompt_parts.append(f"Assistant: {m.content}")

        full_prompt = "\n\n".join(prompt_parts)
        return full_prompt, system_instruction, images

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt, sys_inst, images = self._extract_inputs(messages)
        text = "".join(
            chunk.message.content
            for chunk in self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt, sys_inst, images = self._extract_inputs(messages)
        with self._lock:
            for piece in self._client.stream(
                prompt=prompt,
                system_instruction=sys_inst,
                images=images if images else None,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            ):
                if piece:
                    if run_manager:
                        run_manager.on_llm_new_token(piece)
                    yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
