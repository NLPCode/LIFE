import unicodedata
def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def maybe_format(page, evidence):
    return f"title: {clean(page)} context: {clean(evidence)}"

def deduplicate(evidence):
    unique = set(map(lambda ev: (ev["page"], ev["line"]), evidence))
    return unique

def clean(page):
    return (
        page.replace("_", " ")
        .replace("-LRB-", "(")
        .replace("-RRB-", ")")
        .replace("-COLON-", ":")
    )
