from __future__ import annotations

import shutil
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

from bench.common import build_inputs, select_cases, synchronize

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'profile' / 'round11_case024_ablation' / 'raw'
CASE_NAME = 'case_024_bfloat16_15x255x1x1x256x8'
BLOCK = 4096


@triton.jit
def _store_zero_kernel(y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    zeros = tl.full((BLOCK,), 0.0, tl.float32).to(tl.bfloat16)
    tl.store(y_ptr + offsets, zeros, mask=mask)


@triton.jit
def _copy_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, values, mask=mask)


@triton.jit
def _positive_kernel(x_ptr, y_ptr, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    s = (scale + values * 0.0).to(tl.bfloat16)
    positive = (s * values).to(tl.bfloat16)
    tl.store(y_ptr + offsets, positive, mask=mask)


@triton.jit
def _z_materialize_kernel(x_ptr, y_ptr, n_elements, input_scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    inp = (input_scale + values * 0.0).to(tl.bfloat16)
    z = (inp * values).to(tl.bfloat16)
    tl.store(y_ptr + offsets, z, mask=mask)
    z_mat = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, z_mat, mask=mask)


@triton.jit
def _negative_no_where_kernel(x_ptr, y_ptr, n_elements, alpha, scale, input_scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = (alpha + values * 0.0).to(tl.bfloat16)
    s = (scale + values * 0.0).to(tl.bfloat16)
    inp = (input_scale + values * 0.0).to(tl.bfloat16)
    z = (inp * values).to(tl.bfloat16)
    tl.store(y_ptr + offsets, z, mask=mask)
    z_mat = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    e = tl.exp(z_mat).to(tl.bfloat16)
    m = (e - 1.0).to(tl.bfloat16)
    alpha_scale = (a * s).to(tl.bfloat16)
    negative = (alpha_scale * m).to(tl.bfloat16)
    tl.store(y_ptr + offsets, negative, mask=mask)


@triton.jit
def _current_like_kernel(x_ptr, y_ptr, n_elements, alpha, scale, input_scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = (alpha + values * 0.0).to(tl.bfloat16)
    s = (scale + values * 0.0).to(tl.bfloat16)
    inp = (input_scale + values * 0.0).to(tl.bfloat16)
    z = (inp * values).to(tl.bfloat16)
    tl.store(y_ptr + offsets, z, mask=mask)
    z_mat = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    e = tl.exp(z_mat).to(tl.bfloat16)
    m = (e - 1.0).to(tl.bfloat16)
    alpha_scale = (a * s).to(tl.bfloat16)
    positive = (s * values).to(tl.bfloat16)
    negative = (alpha_scale * m).to(tl.bfloat16)
    out = tl.where(values > 0.0, positive, negative)
    tl.store(y_ptr + offsets, out, mask=mask)


@triton.jit
def _no_materialize_current_like_kernel(x_ptr, y_ptr, n_elements, alpha, scale, input_scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = (alpha + values * 0.0).to(tl.bfloat16)
    s = (scale + values * 0.0).to(tl.bfloat16)
    inp = (input_scale + values * 0.0).to(tl.bfloat16)
    z = (inp * values).to(tl.bfloat16)
    e = tl.exp(z).to(tl.bfloat16)
    m = (e - 1.0).to(tl.bfloat16)
    alpha_scale = (a * s).to(tl.bfloat16)
    positive = (s * values).to(tl.bfloat16)
    negative = (alpha_scale * m).to(tl.bfloat16)
    out = tl.where(values > 0.0, positive, negative)
    tl.store(y_ptr + offsets, out, mask=mask)


def launch(name, x, alpha, scale, input_scale):
    y = x.new_empty(x.shape)
    n = y.numel()
    grid = (triton.cdiv(n, BLOCK),)
    if name == 'store_zero_offset_mask':
        _store_zero_kernel[grid](y, n, BLOCK=BLOCK)
    elif name == 'copy_offset_mask_load_store':
        _copy_kernel[grid](x, y, n, BLOCK=BLOCK)
    elif name == 'positive_scalar_broadcast':
        _positive_kernel[grid](x, y, n, scale, BLOCK=BLOCK)
    elif name == 'z_materialize_only':
        _z_materialize_kernel[grid](x, y, n, input_scale, BLOCK=BLOCK)
    elif name == 'negative_exp_no_where':
        _negative_no_where_kernel[grid](x, y, n, alpha, scale, input_scale, BLOCK=BLOCK, enable_fp_fusion=False)
    elif name == 'current_like_where':
        _current_like_kernel[grid](x, y, n, alpha, scale, input_scale, BLOCK=BLOCK, enable_fp_fusion=False)
    elif name == 'no_materialize_current_like':
        _no_materialize_current_like_kernel[grid](x, y, n, alpha, scale, input_scale, BLOCK=BLOCK, enable_fp_fusion=False)
    else:
        raise ValueError(name)
    return y


def event_time_us(name, x, alpha, scale, input_scale, repeats=20):
    times = []
    with torch.no_grad():
        for _ in range(10):
            y = launch(name, x, alpha, scale, input_scale)
            synchronize()
        for _ in range(repeats):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            y = launch(name, x, alpha, scale, input_scale)
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end) * 1000.0))
    return statistics.median(times), min(times), max(times)


def profile_variant(name, x, alpha, scale, input_scale):
    out_dir = OUT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=True,
        data_simplification=False,
    )
    with torch.no_grad():
        for _ in range(8):
            y = launch(name, x, alpha, scale, input_scale)
            synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir)),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(5):
                y = launch(name, x, alpha, scale, input_scale)
                synchronize()
                prof.step()
    print(f'PROFILED {name} -> {out_dir}', flush=True)


def main():
    torch.npu.set_device(0)
    cases = {c.name: c for c in select_cases('all')}
    inputs = build_inputs(cases[CASE_NAME], torch.device('npu:0'), 20260614)
    x = inputs['x']
    alpha = float(inputs['alpha'])
    scale = float(inputs['scale'])
    input_scale = float(inputs['input_scale'])
    print(f'case={CASE_NAME} shape={tuple(x.shape)} numel={x.numel()} dtype={x.dtype} block={BLOCK}', flush=True)
    print(f'alpha={alpha} scale={scale} input_scale={input_scale}', flush=True)
    variants = [
        'store_zero_offset_mask',
        'copy_offset_mask_load_store',
        'positive_scalar_broadcast',
        'z_materialize_only',
        'no_materialize_current_like',
        'negative_exp_no_where',
        'current_like_where',
    ]
    for name in variants:
        med, mn, mx = event_time_us(name, x, alpha, scale, input_scale)
        print(f'EVENT {name} median_us={med:.3f} min_us={mn:.3f} max_us={mx:.3f}', flush=True)
        profile_variant(name, x, alpha, scale, input_scale)


if __name__ == '__main__':
    main()
