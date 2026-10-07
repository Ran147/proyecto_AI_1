"""Adapter of the movie recommender."""
import re
from difflib import get_close_matches
from typing import Any

import joblib

from ..base import BaseModel
from .schema import MoviesInputs
from .titles import YEAR, display_title, title_keys

# Minimum similarity to accept a misspelled title, and to suggest one.
MATCH_CUTOFF = 0.86
SUGGESTION_CUTOFF = 0.7
# Shorter queries are too ambiguous to match inside a longer title.
MIN_PARTIAL_LENGTH = 4
# Titles named in the spoken sentence.
SPOKEN_TITLES = 5


# Words that introduce the reference movie in a spoken command; the title
# is whatever follows them (after an optional "the movie" / "the film").
# English and Spanish ("una pelicula como jurassic park").
COMMAND_TITLE = re.compile(
    r"\b(?:like|similar to|como|parecidas? a|similar(?:es)? a)\s+"
    r"(?:(?:the (?:movie|film)|la pelicula)\b\s*)?"
)
# Courtesy words said after the title, which are not part of it.
COURTESY = re.compile(
    r"(?:^|\s+)(?:please|thanks|thank you|por favor|gracias)$"
)
# Words of the request itself: text that has them is not a title
# ("i would like you to recommend a movie", "como estas recomiendame una
# pelicula").
NOT_A_TITLE = re.compile(r"\b(?:recom|movie|film|pelicula)")


class Movies(BaseModel):
    """Recommends movies similar to one the user liked."""

    name = "movies"
    title = "Movie recommendation"
    description = (
        "Recommends movies from one the user liked: item-based "
        "collaborative filtering (cosine similarity between movies over "
        "the MovieLens ratings, damped for small overlaps), with a genre "
        "fallback for movies with few ratings. See "
        "training/movies/README.md for the ranking metrics."
    )
    task_type = "recommendation"
    keywords = [
        "recommend a movie",
        "recommend me a movie",
        "movie recommendation",
        "what movie can i watch",
        "what movie should i watch",
        "suggest a movie",
        "i want to watch a movie",
    ]
    # Spanish commands; the answers stay in English.
    keywords_es = [
        "recomiéndame una película",
        "recomendar una película",
        "recomendar película",
        "qué película puedo ver",
        "recomendación de películas",
        "quiero ver una película",
    ]
    input_schema = MoviesInputs
    # A well-known title of the catalog; see artifacts/movies.meta.json.
    default_input = {"title": "Toy Story (1995)", "top_n": 5}

    def inputs_from_command(self, text: str) -> dict[str, Any]:
        """Take the title said after "like" or "similar to".

        "recommend me a movie like toy story" -> {"title": "toy story"}

        Only the text after the matched word counts, without trailing
        courtesy words. Text that is still part of the request ("i would
        like you to recommend a movie") is not a title: the default one
        is used.
        """
        for match in COMMAND_TITLE.finditer(text):
            title = text[match.end():].strip()
            while COURTESY.search(title):
                title = COURTESY.sub("", title).strip()
            if title and not NOT_A_TITLE.search(title):
                return {"title": title}
        return {}

    def load(self) -> None:
        artifact = joblib.load(self.artifact_path())
        self.movies: dict[int, dict[str, Any]] = artifact["movies"]
        self.neighbors: dict[int, list[tuple[int, float]]] = (
            artifact["neighbors"]
        )
        self.title_index: dict[str, int] = artifact["title_index"]
        self.keys = list(self.title_index)
        # Candidates of the genre fallback: movies with enough ratings.
        self.known = [
            (movie_id, frozenset(self.movies[movie_id]["genres"]))
            for movie_id in self.neighbors
        ]

    # ---------- Title lookup ----------
    def _most_rated(self, keys: list[str]) -> int:
        """Among the movies behind `keys`, the one with the most ratings."""
        return max(
            (self.title_index[key] for key in keys),
            key=lambda movie_id: self.movies[movie_id]["n_ratings"],
        )

    def _similar_keys(self, query: str, cutoff: float) -> list[list[str]]:
        """Groups of keys that look like `query`, best group first.

        The query is compared with whole titles and with the beginning of
        longer ones, so 'harry poter' finds 'harry potter and the ...'.
        """
        groups = [
            [key]
            for key in get_close_matches(query, self.keys, n=3, cutoff=cutoff)
        ]
        starts: dict[str, list[str]] = {}
        for key in self.keys:
            if len(key) > len(query) + 1:
                starts.setdefault(key[:len(query) + 1], []).append(key)
        groups += [
            starts[start]
            for start in get_close_matches(query, starts, n=3, cutoff=cutoff)
        ]
        return groups

    def _find(self, text: str) -> int:
        """Return the movie id that best matches what the user wrote."""
        keys = title_keys(text)
        if not keys:
            raise ValueError("say or type the title of a movie")
        for key in keys:
            if key in self.title_index:
                return self.title_index[key]
        # Main name without the year: the first key that has no year.
        query = keys[len(keys) // 2] if YEAR.search(text) else keys[0]
        if len(query) >= MIN_PARTIAL_LENGTH:
            padded = f" {query} "
            partial = [
                key for key in self.keys
                if query in key and padded in f" {key} "
            ]
            if partial:
                return self._most_rated(partial)
        close = self._similar_keys(query, MATCH_CUTOFF)
        if close:
            return self._most_rated(close[0])
        raise ValueError(self._not_found(text, query))

    def _not_found(self, text: str, query: str) -> str:
        suggestions: list[str] = []
        for group in self._similar_keys(query, SUGGESTION_CUTOFF):
            title = self.movies[self._most_rated(group)]["title"]
            if title not in suggestions:
                suggestions.append(title)
        message = f"I could not find the movie '{text.strip()}'"
        if suggestions:
            options = ", ".join(f"'{title}'" for title in suggestions[:3])
            return f"{message}; did you mean {options}?"
        example = self.default_input["title"]
        return f"{message}; try, for example, '{example}'"

    # ---------- Recommendation ----------
    def _by_genre(self, movie_id: int) -> list[tuple[int, float]]:
        """Well-rated movies that share the most genres (Jaccard index)."""
        genres = frozenset(self.movies[movie_id]["genres"])
        scored = []
        for other, other_genres in self.known:
            union = len(genres | other_genres)
            overlap = len(genres & other_genres) / union if union else 0.0
            scored.append(
                (-overlap, -self.movies[other]["bayes_rating"], other)
            )
        return [(other, -overlap) for overlap, _, other in sorted(scored)]

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        movie_id = self._find(inputs["title"])
        top_n = int(inputs["top_n"])
        collaborative = self.neighbors.get(movie_id, [])
        candidates = list(collaborative)
        if len(candidates) < top_n:
            candidates += self._by_genre(movie_id)
        seen = {self.movies[movie_id]["title"]}
        recommendations = []
        for other, score in candidates:
            movie = self.movies[other]
            if other == movie_id or movie["title"] in seen:
                continue
            seen.add(movie["title"])
            recommendations.append({
                "title": str(movie["title"]),
                "genres": [str(genre) for genre in movie["genres"]],
                "score": round(float(score), 4),
            })
            if len(recommendations) == top_n:
                break
        return {
            "title": str(self.movies[movie_id]["title"]),
            "method": "collaborative" if collaborative else "genres",
            "recommendations": recommendations,
        }

    def speak(self, result: dict[str, Any]) -> str:
        liked = display_title(result["title"])
        names = [
            display_title(item["title"])
            for item in result["recommendations"][:SPOKEN_TITLES]
        ]
        if not names:
            return f"I found no recommendations for {liked}."
        listed = names[0] if len(names) == 1 else (
            ", ".join(names[:-1]) + " and " + names[-1]
        )
        if result["method"] == "genres":
            return (
                f"I have few ratings for {liked}; based on its genres, I "
                f"recommend: {listed}."
            )
        return f"If you liked {liked}, I recommend: {listed}."
