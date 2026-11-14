"""Factory klien LLM, termasuk adapter khusus provider IFM."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from config import get_llm_settings


class IFMChatOpenAI(ChatOpenAI):
    """Preserve IFM's nonstandard `reasoning` field across turns."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(
            response,
            generation_info,
        )

        # Recover the original provider response before LangChain
        # discards nonstandard message fields.
        raw_response = response if isinstance(response, dict) else response.model_dump()

        choices = raw_response.get("choices") or []

        for choice, generation in zip(
            choices,
            result.generations, strict=False,
        ):
            raw_message = choice.get("message") or {}

            # Preserve the field even when its value is "".
            if (
                isinstance(generation.message, AIMessage)
                and "reasoning" in raw_message
            ):
                generation.message.additional_kwargs["reasoning"] = (
                    raw_message["reasoning"]
                )

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )

        if generation_chunk is None:
            return None

        # LangChain mengabaikan field non-standar pada delta streaming, jadi
        # `reasoning` harus diambil sendiri dari payload mentah. Tanpa ini,
        # riwayat hasil streaming kehilangan reasoning dan IFM menolak
        # panggilan berikutnya dengan HTTP 400.
        choices = chunk.get("choices") or []

        if not choices:
            return generation_chunk

        delta = choices[0].get("delta") or {}

        if "reasoning" in delta:
            generation_chunk.message.additional_kwargs["reasoning"] = (
                delta["reasoning"] or ""
            )

        return generation_chunk

    def _get_request_payload(
        self,
        input_,
        *,
        stop=None,
        **kwargs,
    ):
        payload = super()._get_request_payload(
            input_,
            stop=stop,
            **kwargs,
        )

        # This adapter targets the Chat Completions payload.
        if "messages" not in payload:
            return payload

        input_messages = self._convert_input(input_).to_messages()

        # Reinsert the exact field into the serialized assistant message.
        for message, serialized in zip(
            input_messages,
            payload["messages"], strict=False,
        ):
            if (
                isinstance(message, AIMessage)
                and "reasoning" in message.additional_kwargs
            ):
                serialized["reasoning"] = (
                    message.additional_kwargs["reasoning"]
                )

        return payload


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Bangun (sekali) klien LLM sesuai konfigurasi provider."""
    settings = get_llm_settings()

    ModelClient = (
        IFMChatOpenAI if settings.provider_name == "ifm" else ChatOpenAI
    )

    return ModelClient(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
