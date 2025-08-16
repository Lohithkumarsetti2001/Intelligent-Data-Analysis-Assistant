
# Super-lightweight code sandbox to evaluate *restricted* pandas snippets.
# WARNING: Still evaluate carefully; keep this minimal.
def run_pandas_snippet(snippet: str, local_vars: dict) -> dict:
    SAFE_GLOBALS = {"__builtins__": {}}
    allowed = {"df","pd","np"}
    safe_locals = {k: v for k,v in local_vars.items() if k in allowed}
    exec(snippet, SAFE_GLOBALS, safe_locals)
    return safe_locals
