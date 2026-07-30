"""Small stdio MCP server used by nooa-coding integration tests."""

from mcp.server import FastMCP

mcp = FastMCP("nooa-coding-test")


@mcp.tool()
def echo(text: str) -> str:
    """Return text from an external MCP server."""
    return f"external:{text}"


@mcp.tool()
def large_output(size: int) -> str:
    """Return a deterministic large payload."""
    return "x" * size


if __name__ == "__main__":
    mcp.run(transport="stdio")
