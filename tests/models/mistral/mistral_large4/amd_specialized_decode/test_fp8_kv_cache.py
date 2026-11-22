# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU tests of the Mistral Large 4 specialized decode on an fp8 latent KV cache.

One TP4 (or TP8) rank's shard of one MoE layer per kind runs on a single GPU
through `Ml4MonoKernel` (npes=1), with random weights, a randomly paged cache
and one DSpark verify step of 4 rows whose last row is padding (slot -1:
KV-sharing layers used to load it ~4 GiB past the source cache, a GPU memory
fault).

The cache holds the same content as bf16 and as fp8 (written by vLLM's fp8
cache writer, which fills it on every non-mono step). KV-sharing layers never
write the cache, so their outputs match exactly; the other layers must store
what vLLM's writer stores for the bf16 rows, saturating at +-448.
"""

import functools

import pytest
import torch

from vllm.platforms import current_platform

FP8 = torch.float8_e4m3fn
HIDDEN, HEADS, KVL, PE, VD, E = 6144, 64, 512, 64, 128, 128
QK = KVL + PE
PAGE, PAGES, S, POS0, TOPK = 64, 64, 4, 3000, 2048


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the ML4 mono kernels need gfx950 (MI355X)"
)


def _randn(*shape, scale=1.0):
    return torch.randn(*shape, device="cuda") * scale


@functools.cache
def build_layer(kind: str, tp: int = 4):
    """(kernel variant, LayerWeights) of one random layer as TP rank 0 of
    `tp` holds it, packed as `runtime.weights.build_moe_layer` packs a loaded
    one."""
    from types import SimpleNamespace as NS

    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.weights.config import (  # noqa: E501
        Mxfp4ScaleLayout,
        Mxfp4WeightLayout,
    )
    from vllm.models.mistral.mistral_large4.amd.specialized_decode.mono.common.weights.layer import (  # noqa: E501
        LayerWeights,
    )
    from vllm.models.mistral.mistral_large4.amd.specialized_decode.runtime import (
        weights,
    )
    from vllm.models.mistral.mistral_large4.amd.specialized_decode.runtime.variants import (  # noqa: E501
        load_variant,
    )

    torch.manual_seed(0)
    q128 = weights._q128
    nope_layer, kv_share, _ = weights.KINDS[kind]
    v = load_variant(nope_layer, kv_share)
    C = v.geometry_config.ml4_tp_config(tp)
    NH = HEADS // tp
    bf, f32 = torch.bfloat16, torch.float32
    t = {}
    t["w_q_b"], t["s_q_b"] = q128(_randn(NH * 128, HIDDEN, scale=HIDDEN**-0.5))
    wkva = _randn(KVL + PE, HIDDEN, scale=HIDDEN**-0.5)
    if nope_layer:
        wkva[KVL:] = 0
    t["w_qkv_a"], t["s_qkv_a"] = q128(wkva * (not kv_share))
    t["g_kv"] = (torch.rand(KVL, device="cuda") + 0.5).to(bf)
    wkvb = _randn(NH, C.nope_dim + VD, KVL, scale=KVL**-0.5)
    w_uk = wkvb[:, : C.nope_dim].transpose(1, 2).reshape(NH * KVL, C.nope_dim)
    t["w_uk"], t["s_uk"] = q128(w_uk, 64)
    t["w_uv"], t["s_uv"] = q128(wkvb[:, C.nope_dim :].reshape(NH * VD, KVL))
    t["w_o"], t["s_o"] = q128(_randn(HIDDEN, NH * VD, scale=(NH * VD) ** -0.5))
    t["g_in"] = t["g_post"] = torch.ones(HIDDEN, dtype=bf, device="cuda")
    t["w_r"] = _randn(E, HIDDEN, scale=HIDDEN**-0.5).to(bf)
    t["bias"] = torch.zeros(E, dtype=f32, device="cuda")
    gc, ih, idim = v.geometry_config, weights.INDEX_HEADS, weights.INDEX_DIM
    prm = torch.zeros(gc.ML4_P_WORDS, dtype=f32, device="cuda")
    prm[gc.ML4_P_SS_S : gc.ML4_P_SS_S + NH] = 1.0
    prm[gc.ML4_P_IQ_W : gc.ML4_P_IQ_W + HIDDEN] = 1.0
    prm[gc.ML4_P_ISS_S : gc.ML4_P_ISS_S + ih] = 1.0
    t["g_q"] = prm
    t["w_index_q"], t["s_index_q"] = q128(_randn(ih * idim, HIDDEN, scale=HIDDEN**-0.5))
    kw = _randn(idim + ih, HIDDEN, scale=HIDDEN**-0.5)
    t["w_index_k"], t["s_index_k"] = q128(kw[:idim])
    t["w_index_w"] = kw[idim:].to(bf).contiguous()
    t["g_index_k"] = torch.ones(idim, dtype=f32, device="cuda")
    t["b_index_k"] = torch.zeros(idim, dtype=f32, device="cuda")
    # routed experts: random MXFP4 values / E8M0 scales in vLLM's AITER a4w4
    # storage (the kernels' ATOM layout); their content is irrelevant here
    u8 = torch.uint8
    for w, s, rows, cols in (("w_ug", "s_ug", 2 * C.inter, HIDDEN),
                             ("w_dn", "s_dn", HIDDEN, C.inter)):  # fmt: skip
        values = torch.randint(0, 256, (E, rows, cols // 2), dtype=u8, device="cuda")
        scales = torch.randint(
            118, 122, (E * rows, cols // 32), dtype=u8, device="cuda"
        )
        t[s] = e8m0_shuffle(scales).view(u8)
        t[w] = values
    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = rocm_aiter_ops.shuffle_weights(t["w_ug"].view(fp4), t["w_dn"].view(fp4))
    t["w_ug"], t["w_dn"] = w13.view(u8), w2.view(u8)
    t["w_ug"].is_shuffled = t["w_dn"].is_shuffled = True
    # shared expert: block fp8, re-quantized to MXFP4 as the runtime does
    linear = lambda n, k: NS(  # noqa: E731
        **dict(zip(("weight", "weight_scale_inv"), q128(_randn(n, k, scale=k**-0.5))))
    )
    shared = NS(
        gate_up_proj=linear(2 * C.inter, HIDDEN), down_proj=linear(HIDDEN, C.inter)
    )
    t.update(weights._shared_expert_mxfp4(shared))
    W = LayerWeights(
        NH, t, C, 0, 1,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=E,
    )  # fmt: skip
    return v, W


def vllm_fp8_rows(rows: torch.Tensor) -> torch.Tensor:
    """[n, 576] latent rows as vLLM's fp8 cache writer stores them (scale 1)."""
    from vllm import _custom_ops as ops

    rows, n = rows.to(torch.bfloat16), rows.shape[0]
    cache = torch.zeros(n, 1, QK, dtype=torch.uint8, device=rows.device)
    slots, one = torch.arange(n, device=rows.device), torch.ones(1, device=rows.device)
    ops.concat_and_cache_mla(rows[:, :KVL], rows[:, KVL:], cache, slots, "fp8", one)
    return cache.view(n, QK).view(FP8)


def make_step():
    """Cache content (e4m3 values), fp8 index cache, one 4-row step whose last
    row is padding."""
    torch.manual_seed(1)
    base = vllm_fp8_rows(_randn(PAGES * PAGE, QK, scale=0.5))
    keys = _randn(PAGES, PAGE * 128, scale=0.5).to(FP8).view(torch.uint8)
    scales = torch.ones(PAGES, PAGE, device="cuda").view(torch.uint8)
    index_cache = torch.cat([keys, scales], 1).view(PAGES, PAGE, 132).contiguous()
    pages = torch.randperm(PAGES, device="cuda", dtype=torch.int32)
    positions = torch.arange(POS0, POS0 + S, device="cuda")
    slots = pages[positions // PAGE].long() * PAGE + positions % PAGE
    slots[-1] = -1  # vLLM's PAD_SLOT_ID
    nkeys = torch.where(slots >= 0, torch.clamp(positions + 1, max=TOPK), 0)
    indptr = torch.cat([nkeys.new_zeros(1), nkeys.cumsum(0)]).int()
    freqs = 10000 ** (-torch.arange(32, device="cuda") / 32)
    ang = torch.arange(PAGES * PAGE, device="cuda")[:, None] * freqs[None]
    step = dict(
        h=_randn(S, HIDDEN).to(torch.bfloat16),
        cur_pos=torch.tensor([POS0], dtype=torch.int32, device="cuda"),
        cos=torch.cos(ang).contiguous(), sin=torch.sin(ang).contiguous(),
        block_tables=pages[None].contiguous(), positions=positions,
        slots=slots, indptr=indptr,
    )  # fmt: skip
    return base, index_cache, step


def run_layer(kind, kv_cache_dtype, kv, index_cache, step, W=None, tp=4):
    v, W0 = build_layer(kind, tp)
    kernel = v.op.Ml4MonoKernel(
        W or W0, S, rank=0, npes=1, group=None, topk=TOPK, launches_per_step=1,
        with_indexer=True, index_max_seq=PAGES * PAGE, index_request_width=S,
        index_page_size=PAGE, index_block_table_stride=PAGES,
        index_cache_dtype="fp8", attention_weight="fp8_block128",
        kv_cache_layout="atom", kv_cache_dtype=kv_cache_dtype,
        native_fp4_mfma=True, rope_dtype="fp32", separate_shared=True,
    )  # fmt: skip
    try:
        out = kernel.forward(
            step["h"], step["cur_pos"], kv, kv, None, step["cos"], step["sin"],
            index_cache=index_cache.clone(), block_tables=step["block_tables"],
            positions=step["positions"], slot_mapping=step["slots"],
            sparse_kv_indptr=step["indptr"],
        )[: S - 1]  # fmt: skip
        torch.accelerator.synchronize()
        assert torch.isfinite(out).all()
        return out.clone()
    finally:
        kernel.close()


def run_both(kind, history_scale=1, W=None, tp=4):
    """The step on the same cache content as bf16 and as fp8:
    (out_bf16, out_fp8, cache_bf16, cache_fp8, cache content, written slots)."""
    base, index_cache, step = make_step()
    base = (base.float() * history_scale).to(FP8)  # power of two: exact
    kv_bf, kv_f8 = base.to(torch.bfloat16), base.clone()
    out_bf = run_layer(kind, "bf16", kv_bf, index_cache, step, W, tp)
    out_f8 = run_layer(kind, "fp8", kv_f8, index_cache, step, W, tp)
    return out_bf, out_f8, kv_bf, kv_f8, base, step["slots"][:-1]


@pytest.mark.parametrize("kind", ("rope", "nope", "nope_kvs"))
# history_scale 4: the old rows dominate attention, so the new rows' fp8
# rounding does not show and every kind matches bf16 exactly
@pytest.mark.parametrize("history_scale", (1, 4))
# TP8: 8 heads and 768 expert columns a rank, so the index-Q tiles run past
# the q_b CTAs and the up/gate stage ends on a partial CTA round
@pytest.mark.parametrize("tp", (4, 8))
@torch.inference_mode()
def test_fp8_cache_matches_bf16(kind, history_scale, tp):
    out_bf, out_f8, kv_bf, kv_f8, base, slots = run_both(kind, history_scale, tp=tp)
    written = torch.zeros(PAGES * PAGE, dtype=torch.bool, device="cuda")
    written[slots] = kind != "nope_kvs"  # KV-sharing layers never write
    u8 = torch.uint8
    assert torch.equal(kv_f8[~written].view(u8), base[~written].view(u8))
    if kind != "nope_kvs":
        new = vllm_fp8_rows(kv_bf[written])
        assert torch.equal(kv_f8[written].view(u8), new.view(u8))
    if kind == "nope_kvs" or history_scale > 1:
        assert torch.equal(out_f8, out_bf)
    else:  # attends to this step's rows rounded to fp8
        rel = (out_f8.float() - out_bf.float()).norm() / out_bf.float().norm()
        assert rel < 0.1, rel


@torch.inference_mode()
def test_index_scores_cover_each_context():
    """Plain decode of S requests of different lengths, one row each: the fused
    indexer scores every key of each row's own context in the launch and no key
    past it, whatever the other rows' lengths."""
    v, W = build_layer("rope")
    base, index_cache, step = make_step()
    ctx = torch.tensor([700, 64, 2049, 1], device="cuda")  # keys, incl. the new row
    need = (ctx + PAGE - 1) // PAGE
    pages = torch.randperm(PAGES, device="cuda", dtype=torch.int32)
    tables = torch.zeros(S, PAGES, dtype=torch.int32, device="cuda")
    for s in range(S):
        start = int(need[:s].sum())
        tables[s, : need[s]] = pages[start : start + int(need[s])]
    positions = ctx - 1
    slots = tables[torch.arange(S), positions // PAGE].long() * PAGE + positions % PAGE
    nkeys = torch.clamp(ctx, max=TOPK)
    indptr = torch.cat([nkeys.new_zeros(1), nkeys.cumsum(0)]).int()
    kernel = v.op.Ml4MonoKernel(
        W, S, rank=0, npes=1, group=None, topk=TOPK, launches_per_step=1,
        with_indexer=True, index_max_seq=PAGES * PAGE, index_request_width=1,
        index_page_size=PAGE, index_block_table_stride=PAGES,
        index_cache_dtype="fp8", attention_weight="fp8_block128",
        kv_cache_layout="atom", kv_cache_dtype="fp8",
        native_fp4_mfma=True, rope_dtype="fp32", separate_shared=True,
    )  # fmt: skip
    try:
        kv = base.clone()
        out = kernel.forward(
            step["h"], step["cur_pos"], kv, kv, None, step["cos"], step["sin"],
            index_cache=index_cache.clone(), block_tables=tables,
            positions=positions, slot_mapping=slots, sparse_kv_indptr=indptr,
        )  # fmt: skip
        torch.accelerator.synchronize()
        assert torch.isfinite(out).all()
        # (score, tag) pairs; a fresh scratch holds tag 0 until a launch writes
        off = kernel.scr_layout["index_scores"]
        n = S * PAGES * PAGE
        tags = kernel.scratch[off : off + n * 8].view(torch.int32)[1::2].view(S, -1)
        live = tags[0, 0].item()
        assert live != 0
        for s in range(S):
            assert (tags[s, : ctx[s]] == live).all(), s
            assert (tags[s, ctx[s] :] == 0).all(), s
    finally:
        kernel.close()


LONG_PAGES = 2048  # 131072 keys: eight 16384-key selector leaves


@functools.cache
def long_caches():
    """Latent cache, fp8 index cache and RoPE tables of LONG_PAGES pages."""
    torch.manual_seed(2)
    base = vllm_fp8_rows(_randn(LONG_PAGES * PAGE, QK, scale=0.5))
    keys = _randn(LONG_PAGES, PAGE * 128, scale=0.5).to(FP8).view(torch.uint8)
    scales = torch.ones(LONG_PAGES, PAGE, device="cuda").view(torch.uint8)
    index_cache = torch.cat([keys, scales], 1).view(LONG_PAGES, PAGE, 132)
    freqs = 10000 ** (-torch.arange(32, device="cuda") / 32)
    ang = torch.arange(LONG_PAGES * PAGE, device="cuda")[:, None] * freqs[None]
    cos, sin = torch.cos(ang).contiguous(), torch.sin(ang).contiguous()
    return base, index_cache.contiguous(), cos, sin


def long_decode(rows, rank=0, npes=1, group=None, indexer_cp=False, width=1):
    """Rope-layer kernel of `rows` decode rows over the long caches, `width`
    rows a request (1: plain decode, `rows`: one verify block)."""
    v, W = build_layer("rope")
    return v.op.Ml4MonoKernel(
        W, rows, rank=rank, npes=npes, group=group, topk=TOPK, launches_per_step=1,
        with_indexer=True, index_max_seq=LONG_PAGES * PAGE, indexer_cp=indexer_cp,
        index_request_width=width,
        index_page_size=PAGE, index_block_table_stride=LONG_PAGES,
        index_cache_dtype="fp8", attention_weight="fp8_block128",
        kv_cache_layout="atom", kv_cache_dtype="fp8",
        native_fp4_mfma=True, rope_dtype="fp32", separate_shared=True,
    )  # fmt: skip


def index_step(kernel, h, ctx, tables, seq=LONG_PAGES * PAGE):
    """One step of len(ctx) requests, a row each: every row's index scores
    (`seq` a row), their mailbox tags and the keys its selector picked."""
    base, index_cache, cos, sin = long_caches()
    rows = len(ctx)
    positions = ctx - 1
    slots = tables[torch.arange(rows), positions // PAGE].long() * PAGE
    slots += positions % PAGE
    nkeys = torch.clamp(ctx, max=TOPK)
    indptr = torch.cat([nkeys.new_zeros(1), nkeys.cumsum(0)]).int()
    kv = base.clone()
    kernel.forward(
        h, torch.tensor([POS0], dtype=torch.int32, device="cuda"), kv, kv, None,
        cos, sin, index_cache=index_cache.clone(), block_tables=tables,
        positions=positions, slot_mapping=slots, sparse_kv_indptr=indptr,
    )  # fmt: skip
    torch.accelerator.synchronize()
    off = kernel.scr_layout["index_scores"]
    pairs = kernel.scratch[off : off + rows * seq * 8].view(torch.int32)
    off = kernel.scr_layout["indices"]
    picked = kernel.scratch[off : off + rows * TOPK * 4].view(torch.int32)
    pairs = pairs.view(rows, seq, 2).clone()
    return pairs[..., 0].view(torch.float32), pairs[..., 1], picked.view(rows, -1)


@torch.inference_mode()
def test_index_select_long_contexts():
    """Plain decode of S requests with contexts past one 16384-key selector
    leaf: each CTA scores several key tiles of one row, and the selector merges
    the leaves' lists. Each row's scores cover its own context and equal those
    of the same request decoded alone; its picked keys are its TOPK best."""
    ctx = torch.tensor([40000, 16385, 3000, 20000], device="cuda")
    need = (ctx + PAGE - 1) // PAGE
    pages = torch.randperm(LONG_PAGES, device="cuda", dtype=torch.int32)
    tables = torch.zeros(S, LONG_PAGES, dtype=torch.int32, device="cuda")
    for s in range(S):
        start = int(need[:s].sum())
        tables[s, : need[s]] = pages[start : start + int(need[s])]
    h = _randn(S, HIDDEN).to(torch.bfloat16)
    with long_decode(S) as kernel:
        scores, tags, picked = index_step(kernel, h, ctx, tables)
        picked = picked.long()
    live = tags[0, 0].item()
    assert live != 0
    with long_decode(1) as kernel:
        for s in range(S):
            n = int(ctx[s])
            assert (tags[s, :n] == live).all(), s
            assert (tags[s, n:] == 0).all(), s
            own = scores[s, :n]
            alone = index_step(kernel, h[s : s + 1], ctx[s : s + 1], tables[s : s + 1])
            assert torch.equal(own, alone[0][0, :n]), s
            assert picked[s].unique().numel() == TOPK, s
            assert int(picked[s].max()) < n, s
            kth = own.topk(TOPK).values[-1]
            assert (own[picked[s]] >= kth).all(), s
            assert torch.isin((own > kth).nonzero().flatten(), picked[s]).all(), s


@pytest.mark.parametrize("n", (131072, 40000, 3000))
@torch.inference_mode()
def test_index_verify_block_one_key_load(n, monkeypatch):
    """One request's 4-row verify block (consecutive positions, one block
    table), up to eight selector leaves: scoring all of the block's rows from
    one key load gives every row the scores and picked keys of one task a row
    (MONO_INDEX_QBLOCK=0), and no score past the row's own context."""
    ctx = torch.arange(n - S + 1, n + 1, device="cuda")  # each row's keys
    need = (n + PAGE - 1) // PAGE
    torch.manual_seed(4)
    pages = torch.randperm(LONG_PAGES, device="cuda", dtype=torch.int32)[:need]
    tables = torch.zeros(S, LONG_PAGES, dtype=torch.int32, device="cuda")
    tables[:, :need] = pages
    h = _randn(S, HIDDEN).to(torch.bfloat16)
    got = {}
    for qblock in ("1", "0"):
        monkeypatch.setenv("MONO_INDEX_QBLOCK", qblock)
        with long_decode(S, width=S) as kernel:
            got[qblock] = index_step(kernel, h, ctx, tables)
    (scores, tags, picked), (ref, _, ref_picked) = got["1"], got["0"]
    live = tags[0, 0].item()
    assert live != 0
    for s in range(S):
        m = int(ctx[s])
        assert (tags[s, :m] == live).all(), s
        assert (tags[s, m:] == 0).all(), s
        assert torch.equal(scores[s, :m], ref[s, :m]), s
        assert torch.equal(picked[s].sort().values, ref_picked[s].sort().values), s


@pytest.mark.parametrize("n", (131072, 40000, 3000))
@torch.inference_mode()
def test_index_select_across_leaves(n, monkeypatch):
    """A row's leaves selecting together (MONO_SEL_DIST, no merge level) pick
    the merge tree's keys in the tree's order. The block table repeats 37
    physical pages, so every score recurs on every leaf and the threshold's
    ties split across leaves; this launch's own keys go to an unshared page."""
    ctx = torch.arange(n - S + 1, n + 1, device="cuda")
    need = (n + PAGE - 1) // PAGE
    torch.manual_seed(5)
    pages = torch.randperm(LONG_PAGES, device="cuda", dtype=torch.int32)
    tables = torch.zeros(S, LONG_PAGES, dtype=torch.int32, device="cuda")
    tables[:, : need - 1] = pages[torch.arange(need - 1, device="cuda") % 37]
    tables[:, need - 1] = pages[37]
    h = _randn(S, HIDDEN).to(torch.bfloat16)
    got = {}
    for spread in ("1", "0"):
        monkeypatch.setenv("MONO_SEL_DIST", spread)
        with long_decode(S, width=S) as kernel:
            got[spread] = index_step(kernel, h, ctx, tables)
    (scores, _, picked), (ref, _, ref_picked) = got["1"], got["0"]
    for s in range(S):
        m = int(ctx[s])
        assert torch.equal(scores[s, :m], ref[s, :m]), s
        assert torch.equal(picked[s], ref_picked[s]), s
        kth = scores[s, :m].topk(TOPK).values[-1]
        tied_leaves = ((scores[s, :m] == kth).nonzero().flatten() // 16384).unique()
        assert m <= 16384 or tied_leaves.numel() > 1, s


CP = 4  # TP4: the context-parallel indexer's ranks


def _cp_rank(rank, port, ctx, tables, h, out):
    """One rank of a TP4 step with the context-parallel indexer over the long
    caches: its score plane (its shard of each long row) and picked keys."""
    import torch.distributed as dist

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=CP
    )
    try:
        cp = dict(rank=rank, npes=CP, group=dist.group.WORLD, indexer_cp=True)
        with torch.inference_mode(), long_decode(len(ctx), **cp) as kernel:
            scores, _, picked = index_step(
                kernel, h.cuda(), ctx.cuda(), tables.cuda(), LONG_PAGES * PAGE // CP
            )
        torch.save([scores.cpu(), picked.cpu()], f"{out}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed(num_gpus=CP)
@pytest.mark.skipif(torch.accelerator.device_count() < CP, reason="needs 4 GPUs")
@pytest.mark.parametrize("ctx", [(40000, 16385, 3000, 20000), (40000,), (9000,)])
@torch.inference_mode()
def test_index_select_context_parallel(ctx, tmp_path):
    """TP4 with the context-parallel indexer: each rank scores a contiguous
    quarter of every row's keys and the peers select together. Every rank picks
    the same keys as one GPU, the row's top TOPK by the one-GPU scores, and a
    long row's local shard holds those same scores."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    ctx = torch.tensor(ctx)
    rows, need = len(ctx), (ctx + PAGE - 1) // PAGE
    torch.manual_seed(3)
    pages = torch.randperm(LONG_PAGES, dtype=torch.int32)
    tables = torch.zeros(rows, LONG_PAGES, dtype=torch.int32)
    for s in range(rows):
        start = int(need[:s].sum())
        tables[s, : need[s]] = pages[start : start + int(need[s])]
    h = torch.randn(rows, HIDDEN).to(torch.bfloat16)
    with long_decode(rows) as kernel:
        scores, _, ref = index_step(kernel, h.cuda(), ctx.cuda(), tables.cuda())
    scores, ref = scores.cpu(), ref.cpu().long()
    mp.spawn(_cp_rank, args=(get_open_port(), ctx, tables, h, tmp_path), nprocs=CP)
    for r in range(CP):
        shard, picked = torch.load(tmp_path / f"rank{r}.pt")
        for s in range(rows):
            n, got = int(ctx[s]), picked[s].long()
            own = scores[s, :n]
            kth = own.topk(TOPK).values[-1]
            assert got.unique().numel() == TOPK and int(got.max()) < n, (r, s)
            assert (own[got] >= kth).all(), (r, s)
            assert torch.isin((own > kth).nonzero().flatten(), got).all(), (r, s)
            assert torch.equal(got.sort().values, ref[s].sort().values), (r, s)
            if n > 16384:  # past one selector leaf, a rank keeps its scores
                stride = -(-n // (CP * 64)) * 64  # whole 64-key scoring tiles
                part = own[r * stride : (r + 1) * stride]
                assert torch.equal(shard[s, : part.numel()], part), (r, s)


@pytest.mark.parametrize("kind", ("rope", "nope"))
@torch.inference_mode()
def test_fp8_cache_saturates(kind):
    """Latent and rope key values beyond +-448 saturate, as vLLM's writer does,
    instead of becoming NaN."""
    from dataclasses import replace

    _, W = build_layer(kind)
    t = dict(W.t, g_kv=W.t["g_kv"] * 1000, s_qkv_a=W.t["s_qkv_a"] * 1000)
    _, _, kv_bf, kv_f8, _, slots = run_both(kind, W=replace(W, t=t))
    new_bf, new_f8 = kv_bf[slots], kv_f8[slots]
    assert (new_bf[:, :KVL].float().abs() > 448).any()
    assert (new_bf[:, KVL:].float().abs() > 448).any() == (kind == "rope")
    assert not torch.isnan(new_f8.float()).any()
    assert torch.equal(
        new_f8.view(torch.uint8), vllm_fp8_rows(new_bf).view(torch.uint8)
    )
