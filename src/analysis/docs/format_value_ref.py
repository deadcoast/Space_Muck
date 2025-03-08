def format_value(value: Any) -> str:
    """Format a configuration value for documentation."""
    if isinstance(value, dict):
        items = "\n".join(f"    {k}: {format_value(v)}" for k, v in value.items())
        return f"{{\n{items}\n}}"
    elif isinstance(value, list):
        items = "\n".join(f"    {format_value(v)}" for v in value)
        return f"[\n{items}\n]"
    else:
        return str(value)
