from __future__ import annotations

import csv, glob, shutil, statistics
from pathlib import Path
import torch
import torch_npu
import triton
import triton.language as tl
from bench.common import synchronize

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'profile' / 'round11_manual_loop' / 'raw'
SUMMARY = ROOT / 'profile' / 'round11_manual_loop' / 'analysis' / 'summary.csv'
N = 15 * 255 * 1 * 1 * 256 * 8
BLOCK = 4096
NUM_VEC_CORES = 48
MAX_PERSISTENT_ITERS = triton.cdiv(N, BLOCK * NUM_VEC_CORES)
ALPHA = -90.76096152799968
SCALE = 23.20560155410449
INPUT_SCALE = -62.17956713671173

@triton.jit
def _tile_body(x_ptr, y_ptr, n_elements, tile_id, alpha, scale, input_scale, BLOCK: tl.constexpr):
    offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
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
def _loop_kernel(x_ptr, y_ptr, n_elements, alpha, scale, input_scale, LOOP_TILES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base_tile = pid * LOOP_TILES
    for i in range(LOOP_TILES):
        tile_id = base_tile + i
        _tile_body(x_ptr, y_ptr, n_elements, tile_id, alpha, scale, input_scale, BLOCK)

@triton.jit
def _persistent_kernel(x_ptr, y_ptr, n_elements, alpha, scale, input_scale, NUM_BLOCKS: tl.constexpr, MAX_ITERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    total_tiles = tl.cdiv(n_elements, BLOCK)
    for i in range(MAX_ITERS):
        tile_id = pid + i * NUM_BLOCKS
        active = tile_id < total_tiles
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = (offsets < n_elements) & active
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

def launch_loop(x, loop_tiles: int):
    y = torch.empty_like(x)
    grid = (triton.cdiv(N, BLOCK * loop_tiles),)
    _loop_kernel[grid](x, y, N, ALPHA, SCALE, INPUT_SCALE, LOOP_TILES=loop_tiles, BLOCK=BLOCK, enable_fp_fusion=False)
    return y

def launch_persistent(x):
    y = torch.empty_like(x)
    _persistent_kernel[(NUM_VEC_CORES,)](x, y, N, ALPHA, SCALE, INPUT_SCALE, NUM_BLOCKS=NUM_VEC_CORES, MAX_ITERS=MAX_PERSISTENT_ITERS, BLOCK=BLOCK, enable_fp_fusion=False)
    return y

def launch(kind, x, loop_tiles=1):
    if kind == 'persistent_48': return launch_persistent(x)
    return launch_loop(x, loop_tiles)

def profile_one(tag, kind, x, loop_tiles=1):
    out_dir = OUT / tag
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exp = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=True,
        data_simplification=False,
    )
    times=[]
    with torch.no_grad():
        for _ in range(8): launch(kind,x,loop_tiles); synchronize()
        for _ in range(10):
            st=torch.npu.Event(enable_timing=True); ed=torch.npu.Event(enable_timing=True)
            st.record(); launch(kind,x,loop_tiles); ed.record(); ed.synchronize()
            times.append(float(st.elapsed_time(ed)*1000.0))
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=3,repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out_dir)),
            record_shapes=False, profile_memory=False, with_stack=False, with_modules=False,
            experimental_config=exp,
        ) as prof:
            for _ in range(5): launch(kind,x,loop_tiles); synchronize(); prof.step()
    return statistics.median(times)

def parse(tag):
    paths=glob.glob(str(OUT/tag/'*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
    rows=[]
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                if 'kernel' in r.get('Name',''): rows.append(r)
    def med(col):
        vals=[]
        for r in rows:
            try: vals.append(float(r[col]))
            except Exception: pass
        return statistics.median(vals) if vals else float('nan')
    return {
        'prof_duration_us': med('Duration(us)'),
        'prof_block_num': med('Block Num'),
        'scalar_ratio': med('aiv_scalar_ratio'),
        'scalar_us': med('aiv_scalar_time(us)'),
        'vec_ratio': med('aiv_vec_ratio'),
        'vec_us': med('aiv_vec_time(us)'),
        'mte2_ratio': med('aiv_mte2_ratio'),
        'mte2_us': med('aiv_mte2_time(us)'),
        'mte3_ratio': med('aiv_mte3_ratio'),
        'mte3_us': med('aiv_mte3_time(us)'),
    }

def main():
    torch.npu.set_device(0)
    OUT.mkdir(parents=True, exist_ok=True); SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260710)
    x=torch.randn((N,),device='npu:0',dtype=torch.bfloat16)
    synchronize()
    ref=launch_loop(x,1); synchronize()
    jobs=[]
    for loop_tiles in [1,2,4,8,16,32,40]:
        jobs.append((f'loop{loop_tiles}', 'loop', loop_tiles, triton.cdiv(N, BLOCK*loop_tiles)))
    jobs.append(('persistent_48', 'persistent_48', 1, NUM_VEC_CORES))
    out=[]
    for tag,kind,loop_tiles,grid_blocks in jobs:
        y=launch(kind,x,loop_tiles); synchronize()
        diff=(y.float()-ref.float()).abs().max().item()
        equal=bool(torch.equal(y,ref))
        evt=profile_one(tag,kind,x,loop_tiles)
        m=parse(tag)
        row={'tag':tag,'kind':kind,'loop_tiles':loop_tiles,'launch_blocks':grid_blocks,'event_us':evt,'max_abs_vs_loop1':diff,'exact_equal_vs_loop1':equal,**m}
        out.append(row)
        print('ROW',row,flush=True)
    with open(SUMMARY,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f'WROTE {SUMMARY}', flush=True)

if __name__=='__main__': main()
