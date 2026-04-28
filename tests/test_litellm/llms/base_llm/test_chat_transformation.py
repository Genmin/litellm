"""
Tests for BaseLLMException — message tail-truncation that prevents provider
response bodies (e.g. "Received={raw_response.text}, ...") from leaking
through error_information.error_message into spend logs and other observability sinks.
"""

from litellm.constants import MAX_EXCEPTION_MESSAGE_LENGTH
from litellm.llms.base_llm.chat.transformation import BaseLLMException


class TestBaseLLMExceptionMessageTruncation:
    def test_short_message_passes_through(self):
        msg = "Bad request: missing field 'model'"
        exc = BaseLLMException(status_code=400, message=msg)
        assert exc.message == msg
        assert str(exc) == msg

    def test_message_at_exact_limit_unchanged(self):
        msg = "x" * MAX_EXCEPTION_MESSAGE_LENGTH
        exc = BaseLLMException(status_code=422, message=msg)
        assert exc.message == msg

    def test_long_message_truncated_to_tail(self):
        leak = '"text": "secret generation content"'
        suffix = "Error converting to valid response block=KeyError"
        # leak sits in the head; long filler pushes it past the kept tail window
        msg = "Received={" + leak + ("a" * 6000) + "}, " + suffix
        exc = BaseLLMException(status_code=422, message=msg)

        assert exc.message.startswith("...[truncated ")
        assert exc.message.endswith(suffix)
        assert leak not in exc.message

    def test_truncation_marker_records_dropped_count(self):
        msg = "z" * (MAX_EXCEPTION_MESSAGE_LENGTH + 500)
        exc = BaseLLMException(status_code=500, message=msg)
        assert "...[truncated 500 chars]..." in exc.message

    def test_other_attributes_unaffected(self):
        msg = "y" * (MAX_EXCEPTION_MESSAGE_LENGTH + 100)
        exc = BaseLLMException(
            status_code=503, message=msg, headers={"x-rate-limit": "1"}
        )
        assert exc.status_code == 503
        assert exc.headers == {"x-rate-limit": "1"}

    def test_str_exception_uses_truncated_message(self):
        # str(exc) is what flows into StandardLoggingPayload.error_information.error_message.
        msg = "q" * (MAX_EXCEPTION_MESSAGE_LENGTH + 1000)
        exc = BaseLLMException(status_code=400, message=msg)
        assert str(exc) == exc.message
        assert len(str(exc)) < len(msg)
