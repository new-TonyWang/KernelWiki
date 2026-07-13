---
id: pr-triton-ascend-3
repo: Ascend/triton-ascend
pr: 3
title: 'fix(language): explicitly throw unexpected dtype error for reduction ops'
author: shen-zhaofeng
date: '2025-05-23'
url: https://gitcode.com/Ascend/triton-ascend/merge_requests/3
source_category: upstream-code
architectures:
- ascend910b
tags:
- ai-core
techniques: []
hardware_features:
- ai-core
kernel_types: []
languages:
- python
- triton
- triton-ascend
captured_at: '2026-06-10'
status: closed
inclusion_reason: kernel file changes
changed_paths:
- ascend/examples/generalization_cases/test_argmax.py
- ascend/examples/generalization_cases/test_argmin.py
- ascend/examples/generalization_cases/test_max.py
- ascend/examples/generalization_cases/test_min.py
- ascend/examples/generalization_cases/test_reduce.py
- ascend/examples/generalization_cases/test_sum.py
- ascend/examples/generalization_cases/test_xorsum.py
- setup.py
- triton_patch/python/triton_patch/language/semantic.py
- triton_patch/python/triton_patch/language/standard.py
merge_sha: 9319a71c
artifact_dir: artifacts/prs/triton-ascend/PR-3
---

## Summary

1. sum, xor_sum dtype check
2. reduce dtype check
3. argmax, argmin, max, min dtype check

用例变化：
argmax 仅新增用例
argmin 仅新增用例
max 仅新增用例
min 仅新增用例
reduce 仅新增用例
sum 仅新增用例
xor_sum 仅新增用例

## Problem

fix(language): explicitly throw unexpected dtype error for reduction ops

## Changed Files

- `ascend/examples/generalization_cases/test_argmax.py`
- `ascend/examples/generalization_cases/test_argmin.py`
- `ascend/examples/generalization_cases/test_max.py`
- `ascend/examples/generalization_cases/test_min.py`
- `ascend/examples/generalization_cases/test_reduce.py`
- `ascend/examples/generalization_cases/test_sum.py`
- `ascend/examples/generalization_cases/test_xorsum.py`
- `setup.py`
- `triton_patch/python/triton_patch/language/semantic.py`
- `triton_patch/python/triton_patch/language/standard.py`

