"""Tests for src/lora_proxy.py against two fake vLLM servers. Run: pytest tests/test_lora_proxy.py"""
import asyncio
import json
import os
import sys

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lora_proxy  # noqa: E402


class FakeVLLM:
    """Records what it was asked. `lag` = how many load_lora_adapter calls answer 'No adapter found' first."""

    def __init__(self, name, lag=0, fail_load=False, healthy=True):
        self.name, self.lag, self.fail_load, self.healthy = name, lag, fail_load, healthy
        self.completions, self.loaded, self.unloaded, self.paused = [], [], [], 0
        self.load_calls = 0
        app = web.Application()
        app.router.add_post("/v1/completions", self.completions_h)
        app.router.add_post("/v1/load_lora_adapter", self.load_h)
        app.router.add_post("/v1/unload_lora_adapter", self.unload_h)
        app.router.add_post("/pause", self.pause_h)
        app.router.add_post("/resume", self.pause_h)
        app.router.add_get("/health", self.health_h)
        app.router.add_get("/server_info", self.info_h)
        self.server = TestServer(app)

    async def completions_h(self, req):
        body = await req.json()
        self.completions.append(body)
        await asyncio.sleep(0.02)
        return web.json_response({"served_by": self.name, "choices": [{"token_ids": [1, 2], "logprobs": {"token_logprobs": [0.0, 0.0]}}]})

    async def load_h(self, req):
        self.load_calls += 1
        body = await req.json()
        if self.fail_load:
            return web.Response(status=400, text="LoRA rank 64 is greater than max_lora_rank 32")
        if self.load_calls <= self.lag:
            return web.Response(status=404, text=f"No adapter found for {body['lora_path']}")
        self.loaded.append(body["lora_name"])
        return web.json_response({"ok": True})

    async def unload_h(self, req):
        self.unloaded.append((await req.json())["lora_name"])
        return web.json_response({"ok": True})

    async def pause_h(self, req):
        self.paused += 1
        return web.json_response({"ok": True})

    async def health_h(self, req):
        return web.Response(status=200 if self.healthy else 503)

    async def info_h(self, req):
        return web.json_response({"parallel_config": {"data_parallel_size": 1}, "served_by": self.name})


async def with_proxy(fakes, env_extra=None):
    for f in fakes:
        await f.server.start_server()
    env = {"UPSTREAM_URLS": ",".join(str(f.server.make_url("")) for f in fakes), "HF_TOKEN": "tok",
           "PROXY_LORA_RETRY_S": "0.05", "PROXY_LORA_LOAD_TIMEOUT_S": "5", **(env_extra or {})}
    cfg = lora_proxy.Config(env)
    proxy = TestServer(lora_proxy.make_app(cfg))
    await proxy.start_server()
    return proxy, cfg


async def teardown(proxy, fakes):
    await proxy.close()
    for f in fakes:
        await f.server.close()


def prompt_ids(seed, n=64):
    return [(seed * 7919 + i * 31) % 50000 for i in range(n)]


def test_same_prompt_sticks_and_different_prompts_spread():
    async def run():
        fakes = [FakeVLLM("a"), FakeVLLM("b")]
        proxy, _ = await with_proxy(fakes)
        async with ClientSession() as s:
            # 8 rollouts of the same prompt, sequentially: all on one server
            for _ in range(8):
                async with s.post(proxy.make_url("/v1/completions"), json={"model": "trl-policy-v3", "prompt": prompt_ids(1), "n": 1}) as r:
                    assert r.status == 200
                    assert (await r.json())["served_by"] in ("a", "b")
            counts = [len(f.completions) for f in fakes]
            assert sorted(counts) == [0, 8], counts
            # 16 different prompts, sequentially: spread evenly (no affinity, least-loaded + round robin)
            for i in range(2, 18):
                async with s.post(proxy.make_url("/v1/completions"), json={"model": "trl-policy-v3", "prompt": prompt_ids(i)}) as r:
                    assert r.status == 200
            counts = [len(f.completions) for f in fakes]
            assert sorted(counts) == [8, 16], counts
            # continuation turn (longer prompt sharing the prefix of prompt 1) follows prompt 1's server
            stuck = fakes[0] if len(fakes[0].completions) == 16 else fakes[1]
            before = len(stuck.completions)
            async with s.post(proxy.make_url("/v1/completions"), json={"model": "trl-policy-v3", "prompt": prompt_ids(1) + [5] * 40}) as r:
                assert r.status == 200
                assert r.headers["X-Proxy-Upstream"] == str(fakes.index(stuck))
            assert len(stuck.completions) == before + 1
            async with s.get(proxy.make_url("/proxy/stats")) as r:
                st = await r.json()
                assert st["router"]["affinity"] == 8
        await teardown(proxy, fakes)
    asyncio.run(run())


def test_shared_chat_template_prefix_does_not_count_as_affinity():
    async def run():
        fakes = [FakeVLLM("a"), FakeVLLM("b")]
        proxy, _ = await with_proxy(fakes)
        template = list(range(1000, 1032))  # 32 tokens = 2 whole blocks shared by every prompt, like a system prompt
        async with ClientSession() as s:
            for i in range(1, 21):  # 20 distinct prompts, one rollout each, sequential
                async with s.post(proxy.make_url("/v1/completions"), json={"model": "m", "prompt": template + prompt_ids(i)}) as r:
                    assert r.status == 200
            counts = sorted(len(f.completions) for f in fakes)
            # Round robin once the template block is recognised as shared (it takes two distinct prompts to see its
            # fan-out), not "whoever served the template last".
            assert counts[1] - counts[0] <= 2, counts
            async with s.get(proxy.make_url("/proxy/stats")) as r:
                st = (await r.json())["router"]
                assert st["unmatched"] >= 18 and st["affinity"] <= 2, st
            # ...and a repeat of prompt 7 still finds its home through the distinguishing blocks
            home = [f for f in fakes if any(p["prompt"] == template + prompt_ids(7) for p in f.completions)][0]
            async with s.post(proxy.make_url("/v1/completions"), json={"model": "m", "prompt": template + prompt_ids(7)}) as r:
                assert r.headers["X-Proxy-Upstream"] == str(fakes.index(home))
        await teardown(proxy, fakes)
    asyncio.run(run())


def test_adapter_version_changes_the_prefix_namespace():
    async def run():
        fakes = [FakeVLLM("a"), FakeVLLM("b")]
        proxy, _ = await with_proxy(fakes)
        async with ClientSession() as s:
            for model in ("trl-policy-v1", "trl-policy-v2"):
                async with s.post(proxy.make_url("/v1/completions"), json={"model": model, "prompt": prompt_ids(1)}) as r:
                    assert r.status == 200
            async with s.get(proxy.make_url("/proxy/stats")) as r:
                assert (await r.json())["router"]["affinity"] == 0  # same tokens, new adapter: no match
        await teardown(proxy, fakes)
    asyncio.run(run())


def test_affinity_yields_to_imbalance():
    async def run():
        fakes = [FakeVLLM("a"), FakeVLLM("b")]
        proxy, _ = await with_proxy(fakes, {"PROXY_IMBALANCE": "2"})
        async with ClientSession() as s:
            async with s.post(proxy.make_url("/v1/completions"), json={"model": "m", "prompt": prompt_ids(1)}) as r:
                home = int(r.headers["X-Proxy-Upstream"])
            # fire 12 copies of the sticky prompt concurrently: only imbalance+1-ish can stay home, the rest spill
            rs = await asyncio.gather(*(s.post(proxy.make_url("/v1/completions"), json={"model": "m", "prompt": prompt_ids(1)}) for _ in range(12)))
            homes = sum(1 for r in rs if int(r.headers["X-Proxy-Upstream"]) == home)
            for r in rs:
                r.release()
            assert 0 < homes < 12, homes
            async with s.get(proxy.make_url("/proxy/stats")) as r:
                st = (await r.json())["router"]
                assert st["spilled"] > 0
        await teardown(proxy, fakes)
    asyncio.run(run())


def test_load_lora_broadcasts_and_retries_lagging_mount():
    async def run():
        fakes = [FakeVLLM("a", lag=3), FakeVLLM("b")]
        proxy, _ = await with_proxy(fakes)
        async with ClientSession() as s:
            async with s.post(proxy.make_url("/v1/load_lora_adapter"), json={"lora_name": "trl-policy-v1", "lora_path": "/lora/x/v1"}) as r:
                assert r.status == 200, await r.text()
        assert fakes[0].loaded == ["trl-policy-v1"] and fakes[1].loaded == ["trl-policy-v1"]
        assert fakes[0].load_calls == 4 and fakes[1].load_calls == 1
        await teardown(proxy, fakes)
    asyncio.run(run())


def test_load_lora_is_all_or_nothing():
    async def run():
        fakes = [FakeVLLM("a"), FakeVLLM("b", fail_load=True)]
        proxy, _ = await with_proxy(fakes)
        async with ClientSession() as s:
            async with s.post(proxy.make_url("/v1/load_lora_adapter"), json={"lora_name": "trl-policy-v1", "lora_path": "/lora/x/v1"}) as r:
                assert r.status == 400
                assert "rolled back" in await r.text()
        assert fakes[0].loaded == ["trl-policy-v1"] and fakes[0].unloaded == ["trl-policy-v1"]
        assert fakes[1].loaded == []
        await teardown(proxy, fakes)
    asyncio.run(run())


def test_pause_unload_health_and_forward():
    async def run():
        fakes = [FakeVLLM("a"), FakeVLLM("b")]
        proxy, _ = await with_proxy(fakes)
        async with ClientSession() as s:
            async with s.post(proxy.make_url("/pause?mode=keep")) as r:
                assert r.status == 200
            assert fakes[0].paused == 1 and fakes[1].paused == 1
            async with s.post(proxy.make_url("/v1/unload_lora_adapter"), json={"lora_name": "nope"}) as r:
                assert r.status == 200
            assert fakes[0].unloaded == ["nope"] and fakes[1].unloaded == ["nope"]
            async with s.get(proxy.make_url("/health")) as r:
                assert r.status == 200
            async with s.get(proxy.make_url("/server_info?config_format=json")) as r:
                assert (await r.json())["parallel_config"]["data_parallel_size"] == 1
            fakes[1].healthy = False
            async with s.get(proxy.make_url("/health")) as r:
                assert r.status == 503
                assert "1/2 servers unhealthy" in await r.text()
        await teardown(proxy, fakes)
    asyncio.run(run())


def test_dead_upstream_is_a_502_not_a_hang():
    async def run():
        fakes = [FakeVLLM("a")]
        proxy, _ = await with_proxy(fakes)
        await fakes[0].server.close()
        async with ClientSession() as s:
            async with s.get(proxy.make_url("/health")) as r:
                assert r.status == 502
        await proxy.close()
    asyncio.run(run())
