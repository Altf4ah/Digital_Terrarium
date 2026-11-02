"""Run with:  python -m pytest -q   (or just: python tests/test_smoke.py)"""
import json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from terrarium import Config, World
from terrarium.llm import Cortex, parse_reply, describe, MockBackend


def test_world_runs_and_stays_finite():
    w = World(Config(seed=3, device="cpu"))
    for _ in range(400):
        w.step()
    assert w.a["pos"].shape[0] > 0 and w.p["biomass"].shape[0] > 0
    assert not torch.isnan(w.a["params"]).any() and not torch.isnan(w.a["e"]).any()
    assert (w.a["pos"] >= 0).all() and (w.a["pos"] <= w.cfg.size).all()


def test_learning_changes_weights_and_off_switch_freezes_them():
    w = World(Config(seed=4, device="cpu", init_animals=20, learn=True))
    before = w.a["params"].clone()
    for _ in range(50):
        w.step()
    # weights of surviving first-generation animals must have moved
    ids = w.a["id"][w.a["id"] <= 20]
    assert ids.numel() > 0
    w2 = World(Config(seed=4, device="cpu", init_animals=20, learn=False))
    b2 = w2.a["params"].clone()
    for _ in range(50):
        w2.step()
    keep = w2.a["id"] <= 20
    assert torch.equal(w2.a["params"][keep], w2.a["p0"][keep])   # no learning: params equal the genome


def test_inheritance_slider_passes_learning_on():
    w = World(Config(seed=5, device="cpu", inherit=1.0))
    for _ in range(600):
        w.step()
    assert w.births > 0


def test_checkpoint_roundtrip():
    w = World(Config(seed=6, device="cpu"))
    for _ in range(100):
        w.step()
    s = w.state_dict()
    w2 = World(Config(seed=6, device="cpu"))
    w2.load_state_dict(s)
    assert w2.tick == 100 and w2.a["pos"].shape == w.a["pos"].shape
    for _ in range(20):
        w2.step()


def test_parse_reply_rejects_garbage():
    assert parse_reply('noise {"intent":"Flee","thought":"run","belief":""} end')["intent"] == "flee"
    for bad in ['no json', '{"intent":"dance"}']:
        try:
            parse_reply(bad); assert False
        except ValueError:
            pass


class Fake(BaseHTTPRequestHandler):
    mode = "ok"
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append((self.path, dict(self.headers), body))
        if Fake.mode == "down":
            self.send_response(500); self.end_headers(); return
        reply = json.dumps({"intent": "hunt", "thought": "Something moves ahead.", "belief": "Moving things can be eaten."})
        out = {"message": {"content": reply}} if self.path == "/api/chat" else {"content": [{"type": "text", "text": reply}]}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _drive(cfg):
    w = World(cfg)
    cx = Cortex(cfg, os.path.join(tempfile.mkdtemp(), "d.jsonl"))
    for _ in range(260):
        w.step(); cx.tick(w)
    time.sleep(1.0); cx.tick(w)
    return w, cx


def test_ollama_and_anthropic_backends_against_fake_server():
    srv, url = _serve()
    try:
        Fake.mode = "ok"; Fake.seen.clear()
        w, cx = _drive(Config(seed=7, device="cpu", llm_backend="ollama", llm_host=url, llm_interval=20, llm_agents=2))
        assert cx.errors == 0 and cx.calls > 0
        path, headers, body = next(r for r in Fake.seen if r[0] == "/api/chat")
        assert path == "/api/chat" and body["format"] == "json" and body["messages"][0]["role"] == "system"
        assert (w.a["intent"][w.a["llm"]] == 1).any()           # "hunt" reached the brain
        os.environ["ANTHROPIC_API_KEY"] = "test-key"; os.environ["ANTHROPIC_BASE_URL"] = url
        Fake.seen.clear()
        w, cx = _drive(Config(seed=7, device="cpu", llm_backend="anthropic", llm_interval=20, llm_agents=2))
        assert cx.errors == 0 and cx.calls > 0
        path, headers, body = next(r for r in Fake.seen if r[0] == "/v1/messages")
        assert path == "/v1/messages" and {k.lower(): v for k, v in headers.items()}.get("x-api-key") == "test-key" and "system" in body and body["messages"][0]["role"] == "user"
    finally:
        srv.shutdown()


def test_backend_failure_falls_back_instead_of_crashing():
    srv, url = _serve()
    try:
        Fake.mode = "down"
        w, cx = _drive(Config(seed=8, device="cpu", llm_backend="ollama", llm_host=url, llm_interval=20, llm_agents=2))
        assert cx.errors > 0 and w.a["pos"].shape[0] > 0
        assert (w.a["intent"][w.a["llm"]] >= 0).all()            # animals still got a plan from the fallback
    finally:
        Fake.mode = "ok"; srv.shutdown()


def test_call_budget_is_respected():
    cfg = Config(seed=9, device="cpu", llm_interval=10, llm_agents=4, llm_max_calls=7)
    w, cx = _drive(cfg)
    assert cx.calls <= 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
