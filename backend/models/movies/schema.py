"""Input schema of the movie recommender."""
from pydantic import BaseModel as Schema
from pydantic import Field

MAX_RECOMMENDATIONS = 20


class MoviesInputs(Schema):
    """A movie the user liked and how many recommendations to return."""

    title: str = Field(
        min_length=1,
        max_length=200,
        description=(
            "title of a movie the user liked, written as in MovieLens; "
            "case, accents, the year and the article order are optional, "
            "e.g. 'Toy Story (1995)', 'the matrix', 'Matrix, The'"
        ),
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=MAX_RECOMMENDATIONS,
        description="number of movies to recommend, 1 to 20",
    )
