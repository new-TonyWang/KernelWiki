# KernelWiki MCP Token 增删改查用法

本文档说明远程 HTTP MCP 服务的 bearer token 管理方式。推荐使用
SQLite token DB 模式，这样 MCP 服务启动后，可以继续新增、禁用、修改、
轮换和删除 token，且变更立即生效，无需重启服务。

## 1. 启用动态 token DB

先创建第一个客户端 token：

```bash
python3 scripts/mcp_token_admin.py --db data/mcp_tokens.sqlite3 add laptop
```

命令会打印 token 明文。请立刻复制保存；后续数据库只保存 salted PBKDF2
hash，不能再次查看明文。

启动 HTTP MCP 服务：

```bash
MCP_TOKEN_DB=data/mcp_tokens.sqlite3 \
MCP_ADMIN_TOKEN='replace-with-a-long-random-admin-token' \
python3 scripts/mcp_http_server.py --host 0.0.0.0 --port 8765
```

客户端使用刚才创建的 token：

```bash
export KERNELWIKI_MCP_TOKEN='<token printed by add/rotate>'
codex mcp add kernelwiki-remote \
  --url http://SERVER_HOST:8765/mcp \
  --bearer-token-env-var KERNELWIKI_MCP_TOKEN
```

## 2. CLI 增删改查

所有 CLI 命令都通过 `--db` 指定同一个 SQLite 数据库：

```bash
DB=data/mcp_tokens.sqlite3
```

### 增：新增 token

自动生成 token：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" add ci-runner --note "CI access"
```

使用指定 token 明文：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" add local-dev --token 'my-secret-token'
```

创建时先禁用：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" add temp-client --disabled
```

### 查：列出或查看 token

列出所有 token，不显示明文：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" list
```

只列出启用的 token：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" list --active-only
```

查看单个 token：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" get 1
```

JSON 输出：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" --json list
```

### 改：修改、启用、禁用、轮换 token

修改名称和备注：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" update 1 \
  --name laptop-new \
  --note "renamed laptop token"
```

禁用 token，正在运行的 MCP 服务会立即拒绝该 token：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" disable 1
```

重新启用：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" enable 1
```

轮换 token secret，旧 token 立即失效，新 token 明文只打印一次：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" rotate 1
```

### 删：删除 token

交互确认删除：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" delete 1
```

跳过确认：

```bash
python3 scripts/mcp_token_admin.py --db "$DB" delete 1 -y
```

删除后，正在运行的 MCP 服务会立即拒绝该 token。

## 3. HTTP Admin API 增删改查

HTTP Admin API 需要服务端设置 `MCP_ADMIN_TOKEN`。下面假设：

```bash
ADMIN='replace-with-a-long-random-admin-token'
BASE='http://SERVER_HOST:8765'
```

### 增：新增 token

```bash
curl -sS -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"ci-runner","note":"CI access"}' \
  "$BASE/admin/tokens"
```

响应里的 `token.token` 是新 token 明文，只会返回这一次。

### 查：列出或查看 token

列出：

```bash
curl -sS -H "Authorization: Bearer $ADMIN" \
  "$BASE/admin/tokens"
```

查看单个：

```bash
curl -sS -H "Authorization: Bearer $ADMIN" \
  "$BASE/admin/tokens/1"
```

### 改：修改、禁用、启用、轮换

修改名称、备注或启用状态：

```bash
curl -sS -X PATCH -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"name":"laptop-new","note":"renamed","enabled":true}' \
  "$BASE/admin/tokens/1"
```

禁用：

```bash
curl -sS -X PATCH -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false}' \
  "$BASE/admin/tokens/1"
```

启用：

```bash
curl -sS -X PATCH -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{"enabled":true}' \
  "$BASE/admin/tokens/1"
```

轮换 token secret：

```bash
curl -sS -H "Authorization: Bearer $ADMIN" \
  -H 'Content-Type: application/json' \
  -d '{}' \
  "$BASE/admin/tokens/1/rotate"
```

### 删：删除 token

```bash
curl -sS -X DELETE -H "Authorization: Bearer $ADMIN" \
  "$BASE/admin/tokens/1"
```

## 4. 验证 token 是否生效

无 token 请求应返回 401：

```bash
curl -i -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' \
  "$BASE/mcp"
```

带有效 token 请求应返回 JSON-RPC 成功响应：

```bash
TOKEN='<client-token>'
curl -sS -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' \
  "$BASE/mcp"
```

预期：

```json
{"jsonrpc":"2.0","id":1,"result":{}}
```

## 5. 安全说明

- `data/mcp_tokens*.sqlite*` 已加入 `.gitignore`，不要把 token DB 提交到仓库。
- token 明文只在 `add` / `rotate` 时返回一次；`list` / `get` 不会显示明文。
- `MCP_ADMIN_TOKEN` 权限很高，应和客户端 token 分开保存。
- 生产环境建议放到 HTTPS 反向代理后面，避免明文 HTTP 传输 bearer token。
- 如果同时设置 `MCP_TOKEN_DB` 和 `MCP_AUTH_TOKEN`，服务会把
  `MCP_AUTH_TOKEN` 作为 `env-bootstrap` 插入 DB 一次，便于从静态 token
  迁移到动态 token DB。

## 6. 自动化测试

```bash
bash scripts/test_mcp_http_smoke.sh
```

该测试覆盖：

- HTTP MCP 基础调用
- 静态 bearer token 鉴权
- SQLite token DB 鉴权
- CLI 禁用/启用运行中服务的 token
- HTTP Admin API 的新增、查询、修改、轮换和删除
