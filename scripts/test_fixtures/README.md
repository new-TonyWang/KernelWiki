# MCP Smoke Test Fixtures

Each `.json` fixture contains a JSON array of test cases. Each test case has:
- `name`: descriptive test name
- `request`: the JSON-RPC 2.0 message to send
- `checks`: list of assertions, each with `path` (jq-style), `op` (eq/neq/contains/exists), and `expected`
- `expect_no_response` (optional): true if the message should produce no response (notifications)

The test runner sends all requests in a single session to verify server-stays-alive behavior.
