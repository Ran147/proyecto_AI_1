"""Trained ``movies`` recommender.

Tests that run the model are skipped when artifacts/movies.joblib is
missing (training/movies/notebook.ipynb generates it).
"""
import json

import joblib
import pytest
from fastapi.testclient import TestClient

from backend.core.errors import ModelUnavailableError
from backend.core.orchestrator import ModelOrchestrator, get_orchestrator
from backend.core.registry import ModelRegistry
from backend.main import app
from backend.models.movies import Movies
from backend.models.movies.titles import display_title, title_keys

ARTIFACT = Movies().artifact_path()
META = Movies().artifact_path(".meta.json")

needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason=f"{ARTIFACT.name} missing; run training/movies/notebook.ipynb",
)
client = TestClient(app)


def post(inputs=None):
    return client.post("/models/movies/predict", json={"inputs": inputs or {}})


def predict(inputs=None):
    response = post(inputs)
    assert response.status_code == 200
    return response.json()


def test_registered_as_real_model():
    info = get_orchestrator().describe("movies")
    assert info["demo"] is False
    assert info["task_type"] == "recommendation"
    assert set(info["input_fields"]) == set(Movies.default_input)


def test_missing_artifact_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        Movies,
        "artifact_path",
        lambda self, suffix=".joblib": tmp_path / "missing.joblib",
    )
    orchestrator = ModelOrchestrator(ModelRegistry.from_classes([Movies]))
    with pytest.raises(ModelUnavailableError):
        orchestrator.predict("movies")


def test_invalid_input_rejected_before_loading():
    for inputs in (
        {"top_n": 0},
        {"top_n": 21},
        {"top_n": "many"},
        {"title": ""},
        {"title": 1995},
    ):
        assert post(inputs).status_code == 422


def test_title_keys_ignore_case_accents_year_and_article_order():
    assert title_keys("Matrix, The (1999)") == ["matrix 1999", "matrix"]
    assert title_keys("the MATRIX") == ["matrix"]
    assert title_keys("Amélie") == title_keys("amelie")
    assert "se7en" in title_keys("Seven (a.k.a. Se7en) (1995)")
    assert title_keys("  ") == []
    assert display_title("Matrix, The (1999)") == "The Matrix"
    assert display_title("Seven (a.k.a. Se7en) (1995)") == "Seven"


@needs_artifact
def test_predict_default_input():
    body = predict()
    json.dumps(body)  # must be serializable
    result = body["result"]
    top_n = Movies.default_input["top_n"]
    assert result["title"] == Movies.default_input["title"]
    assert result["method"] == "collaborative"
    titles = [item["title"] for item in result["recommendations"]]
    assert len(titles) == top_n
    assert len(set(titles)) == top_n
    assert result["title"] not in titles
    scores = [item["score"] for item in result["recommendations"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < score <= 1 for score in scores)
    assert all(
        isinstance(item["genres"], list) for item in result["recommendations"]
    )


@needs_artifact
def test_default_recommendations_match_the_notebook():
    expected = json.loads(META.read_text())["default_recommendations"]
    got = predict()["result"]["recommendations"]
    assert [item["title"] for item in got] == [
        item["title"] for item in expected
    ]


@needs_artifact
@pytest.mark.parametrize("top_n", [1, 3, 20])
def test_returns_exactly_top_n_distinct_movies(top_n):
    result = predict({"top_n": top_n})["result"]
    titles = [item["title"] for item in result["recommendations"]]
    assert len(titles) == len(set(titles)) == top_n
    assert result["title"] not in titles


@needs_artifact
@pytest.mark.parametrize("title", [
    "Matrix, The (1999)",
    "the matrix",
    "THE MATRIX (1999)",
    "Matrix, The",
    "matrix",
    "  The Mátrix  ",
])
def test_title_matching_is_tolerant(title):
    assert predict({"title": title})["result"]["title"] == "Matrix, The (1999)"


@needs_artifact
def test_partial_and_misspelled_titles_are_matched():
    def matched(title):
        return predict({"title": title})["result"]["title"]

    assert matched("toy stori") == "Toy Story (1995)"
    assert matched("se7en") == "Seven (a.k.a. Se7en) (1995)"
    assert matched("star wars").startswith("Star Wars")


@needs_artifact
def test_unknown_title_is_rejected_with_an_english_message():
    response = post({"title": "zzzz qqqq xxxx"})
    assert response.status_code == 422
    body = json.dumps(response.json(), ensure_ascii=False)
    assert "I could not find the movie 'zzzz qqqq xxxx'" in body
    assert "try, for example, 'Toy Story (1995)'" in body
    blank = post({"title": "   "})
    assert blank.status_code == 422
    assert "title of a movie" in json.dumps(blank.json())


@needs_artifact
def test_unknown_title_suggests_close_titles():
    # Too far from the title to be matched, close enough to be suggested.
    response = post({"title": "jurasik parc"})
    assert response.status_code == 422
    body = json.dumps(response.json(), ensure_ascii=False)
    assert "I could not find the movie 'jurasik parc'" in body
    assert "did you mean 'Jurassic Park (1993)'?" in body


@needs_artifact
def test_movie_with_few_ratings_falls_back_to_genres():
    artifact = joblib.load(ARTIFACT)
    movie = next(
        movie for movie_id, movie in artifact["movies"].items()
        if movie_id not in artifact["neighbors"] and movie["genres"]
    )
    result = predict({"title": movie["title"], "top_n": 3})["result"]
    assert result["method"] == "genres"
    assert len(result["recommendations"]) == 3


@needs_artifact
def test_title_index_agrees_with_the_adapter_normalisation():
    # The notebook builds the index with its own copy of title_keys.
    artifact = joblib.load(ARTIFACT)
    for movie in artifact["movies"].values():
        for key in title_keys(movie["title"]):
            assert key in artifact["title_index"], movie["title"]


def test_keywords_are_the_english_voice_commands():
    assert Movies.keywords == [
        "recommend a movie",
        "recommend me a movie",
        "movie recommendation",
        "what movie can i watch",
        "what movie should i watch",
        "suggest a movie",
        "i want to watch a movie",
    ]


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, recommend a movie",
    "Jarvis tec recommend me a movie",
    "JarvisTEC movie recommendation",
    "JarvisTEC, what movie can I watch?",
    "JarvisTEC what movie should I watch?",
    "Jarvis tec suggest a movie",
    "JarvisTEC, I want to watch a movie",
])
def test_voice_command_runs_model(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "movies"
    assert len(body["result"]["recommendations"]) == 5
    assert body["message"].startswith("If you liked Toy Story, I recommend: ")
    assert " and " in body["message"]


@pytest.mark.parametrize(("text", "title"), [
    ("recommend me a movie like jurassic park", "jurassic park"),
    ("recommend a movie similar to the matrix", "the matrix"),
    ("recommend a movie like the movie jurassic park", "jurassic park"),
    ("suggest a movie similar to the film titanic", "titanic"),
    ("recommend a movie like jurassic park please", "jurassic park"),
    ("recommend a movie like jurassic park thanks", "jurassic park"),
    ("recommend a movie like jurassic park thank you", "jurassic park"),
    (
        "recommend a movie similar to jurassic park please thank you",
        "jurassic park",
    ),
    # The title is what follows the LAST usable "like".
    (
        "i would like you to recommend a movie like jurassic park",
        "jurassic park",
    ),
    ("i want to watch a movie like some like it hot", "some like it hot"),
])
def test_title_is_taken_from_the_command(text, title):
    assert Movies().inputs_from_command(text) == {"title": title}


@pytest.mark.parametrize("text", [
    "recommend me a movie",
    "what movie can i watch",
    # "like" is part of the request, not the introduction of a title.
    "i would like you to recommend a movie",
    "i would like a movie recommendation",
    "i would like to watch a film",
    # Nothing left once the courtesy words are removed.
    "recommend a movie like please",
    "recommend a movie similar to the movie thank you",
    # No title follows the word.
    "recommend a movie i like",
])
def test_command_without_a_title_keeps_the_default(text):
    assert Movies().inputs_from_command(text) == {}


@needs_artifact
@pytest.mark.parametrize("text", [
    "JarvisTEC, recommend me a movie like Jurassic Park",
    "JarvisTEC recommend a movie similar to Jurassic Park, please",
    "Jarvis tec suggest a movie like the movie Jurassic Park. Thank you!",
])
def test_voice_command_with_a_title(text):
    body = client.post("/command", json={"text": text}).json()
    assert body["matched"]
    assert body["model"] == "movies"
    assert body["result"]["title"] == "Jurassic Park (1993)"
    assert body["message"].startswith(
        "If you liked Jurassic Park, I recommend: "
    )


@needs_artifact
def test_request_that_only_says_like_uses_the_default_title():
    body = client.post(
        "/command",
        json={"text": "JarvisTEC, I would like you to recommend a movie"},
    ).json()
    assert body["matched"]
    assert body["model"] == "movies"
    assert body["result"]["title"] == Movies.default_input["title"]
    assert "If you liked Toy Story" in body["message"]


@needs_artifact
def test_voice_command_with_an_unknown_title_is_rejected_in_english():
    response = client.post(
        "/command",
        json={"text": "JarvisTEC recommend a movie like zzzz qqqq xxxx"},
    )
    assert response.status_code == 422
    assert "I could not find the movie 'zzzz qqqq xxxx'" in json.dumps(
        response.json(), ensure_ascii=False
    )


def test_spoken_sentence_lists_the_titles_in_english():
    def movie(title):
        return {"title": title, "genres": [], "score": 0.5}

    model = Movies()
    result = {
        "title": "Matrix, The (1999)",
        "method": "collaborative",
        "recommendations": [
            movie("Fight Club (1999)"), movie("Usual Suspects, The (1995)"),
        ],
    }
    assert model.speak(result) == (
        "If you liked The Matrix, I recommend: Fight Club and The Usual "
        "Suspects."
    )
    assert model.speak({**result, "method": "genres"}) == (
        "I have few ratings for The Matrix; based on its genres, I "
        "recommend: Fight Club and The Usual Suspects."
    )
    assert model.speak({**result, "recommendations": []}) == (
        "I found no recommendations for The Matrix."
    )
