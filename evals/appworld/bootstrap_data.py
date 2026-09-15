"""Use upstream unpacking with a checksum-verified official data bundle."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from appworld.download import download_data

payload = Path("/opt/data-0.1.0.bundle").read_bytes()
expected = "fd9f9608c2ec71ed0ac25c3633a738b9129a318a129e31230425b9188e508250"
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError("AppWorld data checksum mismatch")


def cached_get(url, **kwargs):
    if url != "https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle":
        raise RuntimeError("Unexpected download request")
    return SimpleNamespace(status_code=200, content=payload)


# Replace only transport. Official data unpacking and evaluation code are unchanged.
with patch("appworld.download.requests.get", cached_get):
    download_data()
