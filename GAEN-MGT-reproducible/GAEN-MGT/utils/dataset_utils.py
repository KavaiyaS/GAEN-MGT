import re
def clean_text(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\s]", " ", str(t)).lower()
