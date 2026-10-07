"""Title normalisation of the movie recommender.

MovieLens writes titles as ``Matrix, The (1999)`` or
``Seven (a.k.a. Se7en) (1995)``. These functions turn a title, or what a
user typed or said, into lookup keys such as ``matrix`` and ``matrix 1999``.

``normalize``, ``canonical`` and ``title_keys`` are a copy of the functions
of the export cell of training/movies/notebook.ipynb, which builds the
``title_index`` stored in the artifact. Keep both copies identical
(tests/models/test_movies.py checks that they still agree).
"""
import re
import unicodedata

# Articles MovieLens moves to the end of a title ("Matrix, The").
ARTICLES = (
    "the", "a", "an", "la", "le", "les", "el", "los", "las", "il", "lo",
    "das", "der", "die", "un", "une", "una",
)
YEAR = re.compile(r"\((\d{4})(?:-\d{4})?\)\s*$")


def normalize(text: str) -> str:
    """Lowercase, strip accents, keep letters, digits and single spaces."""
    text = unicodedata.normalize("NFD", text.lower().replace("&", " and "))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def canonical(name: str) -> str:
    """Normalised name without its article: 'Matrix, The' -> 'matrix'."""
    words = normalize(name).split()
    if len(words) > 1 and words[-1] in ARTICLES and re.search(
        r",\s*[^\s,]+\s*$", name
    ):
        words = words[:-1]
    elif len(words) > 1 and words[0] in ARTICLES:
        words = words[1:]
    return " ".join(words)


def title_keys(title: str) -> list[str]:
    """Lookup keys of a title; the keys that carry the year come first.

    'Seven (a.k.a. Se7en) (1995)' -> seven 1995, se7en 1995, seven, se7en
    """
    match = YEAR.search(title)
    year = match.group(1) if match else None
    body = YEAR.sub("", title).strip()
    names = [re.sub(r"\([^()]*\)", " ", body)]
    names += re.findall(r"\(([^()]*)\)", body)
    keys = []
    for name in names:
        key = canonical(re.sub(r"^\s*a\.k\.a\.\s*", "", name, flags=re.I))
        if key and key not in keys:
            keys.append(key)
    with_year = [f"{key} {year}" for key in keys] if year else []
    return with_year + keys


def display_title(title: str) -> str:
    """Title as a person says it: 'Matrix, The (1999)' -> 'The Matrix'."""
    name = re.sub(r"\([^()]*\)", " ", title)
    name = " ".join(name.split())
    match = re.match(r"^(.*),\s*([^\s,]+)$", name)
    if match and match.group(2).lower() in ARTICLES:
        name = f"{match.group(2)} {match.group(1)}"
    return name or title
