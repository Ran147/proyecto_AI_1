"""Command routing across the ten registered models.

Every model's own phrases are tested in tests/models/; this file checks the
router as a whole: natural phrasings reach the right model, typos made by a
speech recognizer are forgiven, and phrases about something else reach no
model at all.
"""
import pytest

from backend.core.orchestrator import get_orchestrator
from backend.services.intent_router import strip_wake_word


def route(text: str) -> str | None:
    return get_orchestrator().match(text)[0]


def test_every_keyword_reaches_its_own_model():
    for model in get_orchestrator().list_models():
        assert model["keywords_es"], model["name"]
        for keyword in model["keywords"] + model["keywords_es"]:
            prefixes = ("", "JarvisTEC, ", "hey Jarvis tech ", "oye Jarvis ")
            for prefix in prefixes:
                assert route(prefix + keyword) == model["name"], (
                    prefix + keyword
                )


@pytest.mark.parametrize("text, model", [
    ("JarvisTEC, what is the wine quality", "wine"),
    ("Jarvis, how good is this wine?", "wine"),
    ("JarvisTEC what is the car price today", "car_price"),
    ("Jarvis tec how much is my car worth", "car_price"),
    ("JarvisTEC will the customer leave the company", "churn"),
    ("Jarvis, what is my risk of a stroke", "stroke"),
    ("JarvisTEC what is the avocado price next week", "avocado"),
    ("Jarvis what will the bitcoin price be tomorrow", "bitcoin"),
    ("JarvisTEC how much body fat do I have", "body_fat"),
    ("Jarvis, recommend me a movie like Jurassic Park", "movies"),
    ("JarvisTEC what type of hepatitis does the patient have", "hepatitis"),
    ("Jarvis what is the stage of cirrhosis", "cirrhosis"),
])
def test_natural_phrasings(text, model):
    assert route(text) == model


@pytest.mark.parametrize("text, model", [
    ("JarvisTEC avocato price", "avocado"),
    ("JarvisTEC price of avocadoes", "avocado"),
    ("Jarvis tec bitcoins price", "bitcoin"),
    ("JarvisTEC hepatitus type", "hepatitis"),
    ("JarvisTEC cirrosis stage", "cirrhosis"),
    ("JarvisTEC recommend me a movi", None),  # short word: must be exact
    ("JarvisTEC recommend me a movies", "movies"),
])
def test_recognizer_slips(text, model):
    assert route(text) == model


@pytest.mark.parametrize("text", [
    "JarvisTEC gold price",
    "JarvisTEC price of the bar",
    "JarvisTEC price of the cart",
    "JarvisTEC video quality",
    "JarvisTEC quality of the pine",
    "JarvisTEC risk of a strike",
    "JarvisTEC percentage of grace",
    "JarvisTEC the customer went swimming",
    "JarvisTEC tell me a joke",
    "JarvisTEC what time is it",
    "JarvisTEC",
    "",
])
def test_unrelated_phrases_reach_no_model(text):
    assert route(text) is None


def test_keywords_match_whole_words_only():
    # "wine" inside "swine", "stroke" inside "keystroke".
    assert route("JarvisTEC the swine flu") is None
    assert route("JarvisTEC count every keystroke") is None


@pytest.mark.parametrize("text, rest", [
    ("JarvisTEC, wine quality", "wine quality"),
    ("Jarvis TEC wine quality", "wine quality"),
    ("Jarvis Tech wine quality", "wine quality"),
    ("hey Jarvis wine quality", "wine quality"),
    ("yarvis tek wine quality", "wine quality"),
])
def test_wake_word_variants_are_stripped(text, rest):
    assert strip_wake_word(text) == (rest, True)


def test_command_without_wake_word_is_kept_whole():
    assert strip_wake_word("wine quality") == ("wine quality", False)


# ---------- Spanish commands ----------
@pytest.mark.parametrize("text, model", [
    ("JarvisTEC, calidad del vino", "wine"),
    ("Jarvis, ¿qué tan bueno es este vino?", "wine"),
    ("JarvisTEC cuál es el precio del carro", "car_price"),
    ("Jarvis tec cuánto cuesta mi carro", "car_price"),
    ("JarvisTEC el cliente se va a ir", "churn"),
    ("Jarvis, riesgo de derrame cerebral", "stroke"),
    ("JarvisTEC precio de los aguacates", "avocado"),
    ("Jarvis precio del bitcoin para mañana", "bitcoin"),
    ("JarvisTEC cuánta grasa tengo", "body_fat"),
    ("Jarvis, recomiéndame una película como Jurassic Park", "movies"),
    ("JarvisTEC quiero ver una película", "movies"),
    ("JarvisTEC tipo de hepatitis", "hepatitis"),
    ("Jarvis etapa de cirrosis", "cirrhosis"),
])
def test_spanish_phrasings(text, model):
    assert route(text) == model


@pytest.mark.parametrize("text", [
    "JarvisTEC precio del oro",
    "JarvisTEC precio del barro",
    "JarvisTEC ya vino mi mamá",
    "JarvisTEC el cliente vino a preguntar",
    "JarvisTEC calidad del video",
    "JarvisTEC riesgo de derrumbe",
    "JarvisTEC porcentaje de gracia",
    "JarvisTEC precio del aguacero",
    "JarvisTEC cuéntame un chiste",
])
def test_unrelated_spanish_phrases_reach_no_model(text):
    assert route(text) is None


def test_spanish_data_in_the_sentence():
    orchestrator = get_orchestrator()

    def said(text):
        model, _ = orchestrator.match(text)
        cleaned, _ = strip_wake_word(text)
        return orchestrator._registry.get(model).inputs_from_command(cleaned)

    assert said("JarvisTEC calidad del vino tinto") == {"type": "red"}
    assert said("JarvisTEC calidad del vino blanco") == {"type": "white"}
    assert said("JarvisTEC calidad del vino distinto") == {}
    assert said("JarvisTEC precio del aguacate orgánico") == {
        "type": "organic"
    }
    assert said("JarvisTEC precio del aguacate inorgánico") == {}
    assert said("Jarvis recomiéndame una película como Jurassic Park") == {
        "title": "jurassic park"
    }
    assert said(
        "Jarvis recomiéndame una película parecida a Toy Story por favor"
    ) == {"title": "toy story"}
    # "como" here is not followed by a title.
    assert said("Jarvis como estás recomiéndame una película") == {}
