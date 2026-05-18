"""UDPipe REST API client for Latin text processing."""

from __future__ import annotations

import email.mime.multipart
import email.mime.nonmultipart
import email.policy
import json
import logging
import ssl
import unicodedata
from typing import Any

import pandas as pd

from latin_masking.types import UDPipeAPIError, UDPipeError, UDPipeInputError

logger = logging.getLogger(__name__)

# Default UDPipe service URL
DEFAULT_SERVICE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api"


def _get_ssl_context(unsafe_certs_ok: bool = False) -> ssl.SSLContext:
    """Get SSL context based on unsafe_certs_ok flag.

    Args:
        unsafe_certs_ok: If True, accept self-signed certificates.

    Returns:
        SSL context configured appropriately.

    """
    if unsafe_certs_ok:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context()


# Default SSL context (strict verification)
SSL_CONTEXT = _get_ssl_context(unsafe_certs_ok=True)


def remove_macrons(text: str) -> str:
    """Remove macrons from Latin text.

    Args:
        text: Text to process.

    Returns:
        Text without macrons.

    """
    return unicodedata.normalize("NFD", text).replace("\u0304", "")


def _perform_request(
    method: str,
    params: dict[str, Any] | None,
    service_url: str = DEFAULT_SERVICE_URL,
    unsafe_certs_ok: bool = True,
) -> dict[str, Any]:
    """Perform HTTP request to UDPipe API.

    Args:
        method: API method name (e.g., 'process', 'models').
        params: Request parameters.
        service_url: Base URL for the UDPipe service.
        unsafe_certs_ok: If True, accept self-signed certificates.

    Returns:
        Parsed JSON response.

    Raises:
        UDPipeAPIError: On HTTP or SSL errors.
        UDPipeError: On other errors.

    """
    import urllib.error
    import urllib.request

    if params is None:
        params = {}

    if not params:
        request_headers: dict[str, str] = {}
        request_data: bytes | None = None
    else:
        message = email.mime.multipart.MIMEMultipart(
            "form-data", policy=email.policy.HTTP
        )

        for name, value in params.items():
            payload = email.mime.nonmultipart.MIMENonMultipart("text", "plain")
            payload.add_header("Content-Disposition", f'form-data; name="{name}"')
            payload.add_header("Content-Transfer-Encoding", "8bit")
            payload.set_payload(value, charset="utf-8")
            message.attach(payload)

        request_data = message.as_bytes().split(b"\r\n\r\n", maxsplit=1)[1]
        request_headers = {
            "Content-Type": message["Content-Type"],
            "User-Agent": "If I am generating too much traffic, email benjamin.nagy@ijp.pan.pl",
        }

    ssl_context = _get_ssl_context(unsafe_certs_ok)
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                url=f"{service_url}/{method}",
                headers=request_headers,
                data=request_data,
            ),
            context=ssl_context,
        ) as request:
            try:
                resp = request.read().decode("utf-8")
                result: dict[str, Any] = json.loads(resp)
                return result
            except Exception as e:
                logger.error(
                    "Cannot read the response of UDPipe '%s' REST request: %s",
                    method,
                    repr(e),
                )
                raise UDPipeError(f"Failed to parse response: {e}", e) from e
    except urllib.error.HTTPError as e:
        error_body = e.fp.read().decode("utf-8") if e.fp else ""
        logger.error(
            "UDPipe HTTP error during '%s' request: %s",
            method,
            error_body,
        )
        raise UDPipeAPIError(
            f"HTTP error {e.code}: {error_body}",
            status_code=e.code,
            original_error=e,
        ) from e
    except urllib.error.URLError as e:
        logger.error(
            "SSL/URL error during UDPipe '%s' REST request: %s",
            method,
            repr(e),
        )
        raise UDPipeAPIError(
            f"URL error: {e.reason}",
            original_error=e,
        ) from e
    except json.JSONDecodeError as e:
        logger.error(
            "Cannot parse the JSON response of UDPipe '%s' REST request: %s",
            method,
            e.msg,
        )
        raise UDPipeError(f"JSON decode error: {e.msg}", e) from e


def list_models(service_url: str = DEFAULT_SERVICE_URL) -> list[str]:
    """Query the service and list all available models.

    Args:
        service_url: Base URL for the UDPipe service.

    Returns:
        List of available model names.

    """
    response = _perform_request("models", None, service_url)
    models = response.get("models", [])
    return list(models) if isinstance(models, list) else []


def process_text(
    text: str,
    *,
    model: str = "latin-evalatin24-240520",
    tokenizer: str = "",
    input_type: str = "conllu",
    presegmented: bool = False,
    strip_punct: bool = True,
    remove_macrons_flag: bool = True,
    raw: bool = True,
    service_url: str = DEFAULT_SERVICE_URL,
    unsafe_certs_ok: bool = True,
) -> str | tuple[list[pd.DataFrame], list[str]]:
    """Process text through UDPipe API.

    Args:
        text: Text to process.
        model: UDPipe model name.
        tokenizer: Tokenizer settings.
        input_type: Input type for UDPipe.
        presegmented: Whether text is already pre-segmented (one sentence per line).
        strip_punct: Whether to strip punctuation characters.
        remove_macrons_flag: Whether to remove macrons from input.
        raw: Whether to return raw CoNLL-U response or parsed DataFrames.
        service_url: Base URL for the UDPipe service.
        unsafe_certs_ok: If True, accept self-signed certificates (default: True for UDPipe).

    Returns:
        Raw CoNLL-U string if raw=True, otherwise tuple of (list of DataFrames, list of texts).

    Raises:
        UDPipeInputError: If input text is empty or invalid.
        UDPipeAPIError: On HTTP or API errors.

    """
    if not text or not text.strip():
        raise UDPipeInputError("Input text is empty")

    if strip_punct:
        text = text.translate(str.maketrans("", "", r"[]<>{}†'\""))

    if remove_macrons_flag:
        text = remove_macrons(text)

    data: dict[str, Any] = {
        "input": input_type,
        "output": "conllu",
        "data": text,
        "model": model,
        "tokenizer": tokenizer,
        "parser": "",
        "tagger": "",
    }

    if presegmented:
        # Text is already segmented (one sentence per line)
        tok_arg = data["tokenizer"].split(";")
        tok_arg.append("presegmented")
        data["tokenizer"] = ";".join(tok_arg)
        # Normalize whitespace but preserve line structure
        text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
        data["data"] = text

    response = _perform_request("process", data, service_url, unsafe_certs_ok)

    if "model" not in response or "result" not in response:
        raise UDPipeError("Cannot parse the UDPipe 'process' REST request response.")

    if raw:
        result = response["result"]
        return str(result) if isinstance(result, str) else ""

    # Import here to avoid circular dependency
    from latin_masking.conllu import parse_conllu

    return parse_conllu(response["result"])
