"""Free direct Web reader checks: real official page plus explicit HTTP fixtures."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from tools.local_native import fetch_webpage

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
results = []
client_class = httpx.Client
for status, body in [(200, '<html><body><main>Verified fixture field: 42</main></body></html>'), (403, 'Forbidden'), (429, 'Too many requests'), (200, '<html><script>render()</script><body></body></html>')]:
    response = httpx.Response(status, text=body, headers={"content-type": "text/html; charset=utf-8"})
    def make_client(*a, **kw):
        return client_class(*a, **kw, transport=httpx.MockTransport(lambda request: response))
    with patch("httpx.Client", side_effect=make_client), patch("tools.local_native._public_url", return_value=None):
        try:
            value = fetch_webpage.func("https://fixture.example/page")
            results.append({"fixture": True, "http_status": status, "body": body, "result": value})
        except Exception as e:
            results.append({"fixture": True, "http_status": status, "error_type": type(e).__name__, "error": str(e)})
assert results[0]["result"]["content"].strip() == "Verified fixture field: 42"
assert results[1]["result"]["fetch_status"] == "ACCESS_DENIED"
assert results[2]["result"]["fetch_status"] == "RATE_LIMITED"
assert not results[3]["result"]["content"].strip()
assert results[3]["result"]["fetch_status"] == "EMPTY_CONTENT"
for page in ["csv", "json"]:
    try:
        value = fetch_webpage.func(f"https://docs.python.org/3.12/library/{page}.html", max_length=12000)
        (out / f"live-{page}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({"fixture": False, "page": page, "total_chars": value["total_chars"], "next_index": value["next_index"]})
    except Exception as e:
        results.append({"fixture": False, "page": page, "error": str(e)})
(out / "branches.json").write_text(json.dumps({"model_calls": 0, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False))
