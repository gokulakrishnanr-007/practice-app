import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.html"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "8000"))
MAX_BODY_BYTES = 200_000

# Load GEMINI_API_KEY from .env in the project directory (never sent to frontend).
load_dotenv(BASE_DIR / ".env")


class StructuredResult(BaseModel):
    summary: str = Field(
        description="A concise 1-3 sentence summary of the pasted text."
    )
    key_points: list[str] = Field(
        description="Clear, concise bullet points capturing the key facts, ideas, or decisions."
    )
    action_items: list[str] = Field(
        description="Specific next steps, tasks, or follow-ups mentioned or implied. Empty list if none."
    )


SYSTEM_INSTRUCTION = (
    "You organize messy, unstructured text (notes, emails, chat logs, transcripts). "
    "Extract a concise summary, the main key points, and concrete action items. "
    "Keep each item clear, accurate, and grounded strictly in the provided text."
)


def analyze_with_gemini(text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add GEMINI_API_KEY=your_key to the .env file."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=StructuredResult,
            temperature=0.2,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    parsed = StructuredResult.model_validate_json(response.text)
    return parsed.model_dump()


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # Keep console output clean and avoid logging sensitive request payloads.
        print(f"[{self.address_string()}] {format % args}")

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status_code: int, html_bytes: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html_bytes)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            if not INDEX_PATH.exists():
                self._send_json(404, {"error": "index.html not found."})
                return
            self._send_html(200, INDEX_PATH.read_bytes())
            return

        if self.path == "/api/health":
            self._send_json(200, {"ok": True, "model": MODEL_NAME})
            return

        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self._send_json(404, {"error": "Not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length header."})
            return

        if content_length <= 0:
            self._send_json(400, {"error": "Request body is required."})
            return

        if content_length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "Input text is too large (max 200 KB)."})
            return

        raw_body = self.rfile.read(content_length)
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Invalid JSON request body."})
            return

        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str) or not text.strip():
            self._send_json(400, {"error": "Please paste some text to analyze."})
            return

        try:
            result = analyze_with_gemini(text.strip())
            self._send_json(200, result)
        except RuntimeError as exc:
            self._send_json(500, {"error": str(exc)})
        except genai_errors.APIError as exc:
            status = exc.code if isinstance(exc.code, int) and 400 <= exc.code < 600 else 502
            message = exc.message or "Gemini API request failed."
            self._send_json(status, {"error": f"Gemini API error: {message}"})
        except Exception:
            self._send_json(500, {"error": "Failed to process text with Gemini."})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Server running at http://{HOST}:{PORT} (model: {MODEL_NAME})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
