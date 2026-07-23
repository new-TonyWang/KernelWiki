# KernelWiki MCP 功能与用法

KernelWiki MCP 是面向 **算子优化 / kernel 优化 / GPU-NPU 性能调优** 的知识库查询服务。它把仓库中的 wiki、PR 记录、代码 artifact、source excerpts 暴露给 MCP 客户端，例如 Codex、Claude Code 或其他支持 MCP 的 agent。

AI agent 可以通过这个 MCP：

- 查询 CUDA / PTX / Triton / CuTe / AscendC 算子优化经验
- 查询 NVIDIA SM90 / SM100、Ascend 910B / 910C 的硬件相关技巧
- 查询 CUTLASS、sglang、vLLM、FlashInfer、PyTorch、DeepGEMM 等项目的 PR 记录
- 获取 PR 相关代码、diff、kernel source、benchmark 片段
- 搜索性能症状，例如 memory-bound、register pressure、low SM utilization、pipeline stalls
- 做 operator routing、kernel migration、API definition、code walkthrough 等检索

## 1. 服务形态

当前 MCP 支持两种启动方式。

### 1.1 本地 stdio MCP

适合本机 Codex / Claude Code 直接拉起：

```bash
python3 scripts/mcp_server.py
```

Codex 配置示例：

```bash
codex mcp add kernelwiki -- python3 scripts/mcp_server.py
```

### 1.2 远程 HTTP MCP

适合部署到一台机器，然后其他机器通过 HTTP 访问：

```bash
BLACKWELL_WIKI_ROOT="$PWD" MCP_LOG_FILE=/tmp/kernelwiki-mcp.log \
  python3 scripts/mcp_http_server.py --host 127.0.0.1 --port 8765
```

如果要绑定公网或内网地址，建议开启 token 鉴权：

```bash
MCP_TOKEN_DB=data/mcp_tokens.sqlite3 \
MCP_ADMIN_TOKEN='replace-with-a-long-random-admin-token' \
python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

客户端注册：

```bash
codex mcp add kernelwiki-http --url http://SERVER_HOST:8765/mcp
```

带 bearer token：

```bash
export KERNELWIKI_MCP_TOKEN='<client-token>'
codex mcp add kernelwiki-http \
  --url http://SERVER_HOST:8765/mcp \
  --bearer-token-env-var KERNELWIKI_MCP_TOKEN
```

## 2. MCP 工具列表

### 2.1 `wiki_query`

用途：按关键词和过滤条件查询 KernelWiki。

适合场景：

- 查某个算子、kernel、API 或技术关键词
- 查某个仓库的 PR
- 查有代码 artifact 的页面
- 查某个架构、语言、性能症状相关内容

常用参数：

| 参数 | 说明 | 示例 |
|---|---|---|
| `query` | 关键词数组 | `["flash attention"]`, `["tcgen05"]` |
| `type` | 页面类型 | `pr`, `kernel`, `technique`, `hardware`, `pattern` |
| `repo` | 来源仓库 | `cutlass`, `sglang`, `vllm`, `flashinfer` |
| `language` | 语言 / DSL | `cuda-cpp`, `ptx`, `triton`, `cute-dsl`, `ascendc` |
| `architecture` | 架构 | `sm90`, `sm100`, `ascend910b`, `ascend910c` |
| `symptom` | 性能症状 | `memory-bound`, `register-pressure`, `pipeline-stalls` |
| `has_code` | 只返回带代码 artifact 的页面 | `true` |
| `limit` | 返回数量 | `10` |
| `compact` | 紧凑输出 | `true` |

HTTP 示例：

```bash
curl -sS http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"wiki_query","arguments":{"query":["tcgen05"],"limit":5,"compact":true}}}'
```

### 2.2 `wiki_get_page`

用途：根据 page id 或相对路径获取页面详情。

适合场景：

- 打开 `wiki_query` 查到的页面
- 查看 PR 页面完整内容
- 获取页面关联代码 artifact
- 获取引用 source excerpts

常用参数：

| 参数 | 说明 |
|---|---|
| `lookup` | page id 或相对路径，必填 |
| `body_only` | 只返回 markdown body |
| `frontmatter_only` | 只返回 YAML frontmatter |
| `include_code` | 返回关联代码 artifact |
| `follow_sources` | 返回引用 source excerpts |

HTTP 示例：

```bash
curl -sS http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"wiki_get_page","arguments":{"lookup":"pr-cutlass-3130","include_code":true,"follow_sources":true}}}'
```

### 2.3 `wiki_grep`

用途：在 wiki、source references、code artifacts 中做正则搜索。

适合场景：

- 查某个 API、函数名、kernel 名
- 搜索 PR 代码实现细节
- 搜索 CUDA / PTX / Triton / AscendC 特定写法
- 搜索性能关键词或硬件指令

常用参数：

| 参数 | 说明 | 示例 |
|---|---|---|
| `patterns` | 正则表达式数组，必填 | `["tcgen05"]`, `["wgmma", "sm90"]` |
| `scope` | 搜索范围 | `wiki`, `sources`, `artifacts`, `all` |
| `context` | 上下文行数 | `1`, `2`, `3` |
| `any_match` | 是否任意 pattern 命中即可 | `true` / `false` |
| `limit` | 返回文件数量 | `20` |
| `ext` | 限制扩展名 | `cu,cuh,py,cpp,h` |

HTTP 示例：

```bash
curl -sS http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"wiki_grep","arguments":{"patterns":["tcgen05"],"scope":"all","context":2,"limit":20,"ext":"md,cu,cuh,py,cpp,h"}}}'
```

注意：为了防止批量导出语料，`wiki_grep` 会拒绝 `.`、`.*`、`^` 等 catch-all 正则。请使用具体关键词或通过 `wiki_query` 按 tag/type/repo 查询。

## 3. 算子优化推荐工作流

### 工作流 A：查一个算子的优化资料

1. 用 `wiki_query` 搜索算子或算法名：

```json
{"name":"wiki_query","arguments":{"query":["flash attention"],"type":"kernel","limit":10}}
```

2. 用 `wiki_get_page` 打开候选页面：

```json
{"name":"wiki_get_page","arguments":{"lookup":"kernel-flash-attention-4","include_code":true,"follow_sources":true}}
```

3. 用 `wiki_grep` 搜索实现细节：

```json
{"name":"wiki_grep","arguments":{"patterns":["flash", "attention"],"scope":"artifacts","context":2,"ext":"cu,cuh,py"}}
```

### 工作流 B：查 PR 和 PR 相关代码

1. 查某个 PR：

```json
{"name":"wiki_query","arguments":{"query":["3130"],"type":"pr","repo":"cutlass","limit":5}}
```

2. 打开 PR 页面并带代码：

```json
{"name":"wiki_get_page","arguments":{"lookup":"pr-cutlass-3130","include_code":true,"follow_sources":true}}
```

3. 在 artifact 里搜关键 API：

```json
{"name":"wiki_grep","arguments":{"patterns":["tcgen05"],"scope":"artifacts","context":2,"ext":"cu,cuh,cpp,h"}}
```

### 工作流 C：按性能症状找优化模式

查 memory-bound 相关模式：

```json
{"name":"wiki_query","arguments":{"symptom":"memory-bound","type":"pattern","limit":10}}
```

查 register pressure：

```json
{"name":"wiki_query","arguments":{"symptom":"register-pressure","limit":10}}
```

查 pipeline stalls：

```json
{"name":"wiki_query","arguments":{"symptom":"pipeline-stalls","limit":10}}
```

### 工作流 D：按硬件架构找实现参考

查 Blackwell / SM100：

```json
{"name":"wiki_query","arguments":{"architecture":"sm100","has_code":true,"limit":10}}
```

查 Hopper / SM90：

```json
{"name":"wiki_query","arguments":{"architecture":"sm90","has_code":true,"limit":10}}
```

查 Ascend 910B：

```json
{"name":"wiki_query","arguments":{"architecture":"ascend910b","language":"ascendc","limit":10}}
```

## 4. 自然语言使用示例

在 Codex / Claude 里可以直接这样问：

```text
用 KernelWiki MCP 查 cutlass 里 tcgen05 相关 PR，并打开最相关的页面，包含代码 artifact。
```

```text
我在优化一个 Triton attention kernel，请用 KernelWiki 查相关优化模式、PR 和代码参考。
```

```text
帮我找 AscendC 上 matmul / attention 算子的优化经验，优先返回带代码的页面。
```

```text
在 KernelWiki 的 PR 和代码 artifact 里 grep wgmma.mma_async，显示上下文。
```

```text
查 low SM utilization 相关 pattern，然后给我可能的优化方向和参考代码页面。
```

## 5. Token 管理

远程 HTTP MCP 推荐使用 SQLite token DB。token 增删改查见：

- [`docs/mcp-token-management.md`](mcp-token-management.md)

常用命令：

```bash
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 add laptop
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 list
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 disable 1
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 rotate 1
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 delete 1 -y
```

## 6. 验证 MCP 是否可用

健康检查：

```bash
curl -sS http://127.0.0.1:8765/healthz
```

查看工具列表：

```bash
curl -sS http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

stdio smoke test：

```bash
bash scripts/test_mcp_smoke.sh
```

HTTP smoke test：

```bash
bash scripts/test_mcp_http_smoke.sh
```

## 7. 相关文档

- [`docs/mcp-client-config.md`](mcp-client-config.md)：客户端配置、HTTP 部署、systemd/nginx 示例
- [`docs/mcp-token-management.md`](mcp-token-management.md)：token 增删改查
- [`docs/mcp-validation-checklist.md`](mcp-validation-checklist.md)：验证清单
