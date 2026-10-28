"""Page/timestamp-aware extraction. URLs are fetched only during explicit ingestion."""
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import parse_qs, quote, urljoin, urlsplit

from .models import RagError
from .workspace import validate_url


MAX_RESPONSE = 5 * 1024 * 1024


def public_addresses(host, port):
    addresses = list(dict.fromkeys(info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise RagError("Website URLs must resolve only to public internet addresses.")
    return addresses


class PublicHTTPS(http.client.HTTPSConnection):
    """Pin the connection to a validated address while retaining TLS hostname validation."""
    def connect(self):
        addresses = public_addresses(self.host, self.port)
        last_error = None
        for address in addresses:
            connection = None
            try:
                connection = socket.create_connection((address, self.port), self.timeout)
                self.sock = self._context.wrap_socket(connection, server_hostname=self.host)
                return
            except OSError as error:
                last_error = error
                if connection:
                    connection.close()
        raise last_error or RagError("Cannot connect to any public address for this website.")


def fetch_page(url, include_type=False):
    for _ in range(5):
        url = validate_url(url, "websites")
        parsed = urlsplit(url)
        if parsed.port not in {None, 443}:
            raise RagError("Website ingestion supports only HTTPS port 443.")
        connection = PublicHTTPS(parsed.hostname, timeout=20, context=ssl.create_default_context())
        try:
            target = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
            if parsed.query:
                target += "?" + quote(parsed.query, safe="%=&?/:@!$'()*+,;+-._~")
            connection.request("GET", target, headers={"User-Agent": "SourceLibrary/0.1 (explicit user-requested ingestion)", "Accept-Encoding": "identity"})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise RagError("Website redirect has no destination.")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise RagError(f"Website returned HTTP {response.status}.")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise RagError("Compressed website responses are not supported.")
            content_type = response.getheader("Content-Type", "").split(";")[0].strip().lower()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise RagError("Only HTML and plain-text website pages are supported.")
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise RagError("Website response exceeds 5 MB.")
            return (raw, url, content_type) if include_type else (raw, url)
        finally:
            connection.close()
    raise RagError("Too many website redirects.")


def extract_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted and not reader.decrypt(""):
        raise RagError("Password-protected PDF cannot be read.")
    units, warnings = [], []
    for number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text:
            warnings.append(f"Page {number}: no text extracted; OCR or layout processing may be needed.")
        else:
            units.append({"text": text, "page": number})
    return units, warnings


def extract_website(url):
    import trafilatura

    raw, final, content_type = fetch_page(url, include_type=True)
    text = raw.decode("utf-8-sig", errors="replace").strip() if content_type == "text/plain" else trafilatura.extract(raw, include_tables=True) or ""
    return [{"text": text, "url": final}], []


def extract_youtube(url, languages):
    import requests
    from youtube_transcript_api import YouTubeTranscriptApi

    class TimedSession(requests.Session):
        def request(self, *args, **kwargs):
            kwargs.setdefault("timeout", (5, 30))
            return super().request(*args, **kwargs)

    canonical = validate_url(url, "youtube")
    video = parse_qs(urlsplit(canonical).query)["v"][0]
    with TimedSession() as session:
        session.trust_env = False
        transcript = YouTubeTranscriptApi(http_client=session).fetch(video, languages=languages)
    # Retain adjacent timed segments so chunks cover a useful stretch of speech.
    units, group, start, end = [], [], 0, 0
    for segment in transcript:
        if not group:
            start = segment.start
        group.append(segment.text)
        end = segment.start + segment.duration
        if end - start >= 45:
            units.append({"text": " ".join(group), "start": start, "end": end, "url": canonical})
            group = []
    if group:
        units.append({"text": " ".join(group), "start": start, "end": end, "url": canonical})
    return units, ["Captions may be automatic and inaccurate; visual-only content is not extracted."]
