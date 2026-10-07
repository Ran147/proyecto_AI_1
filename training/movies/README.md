# Movie recommendation (`movies`)

## Problem

- Task: top-N recommendation from one movie the user liked (item-based
  collaborative filtering). It is not supervised learning: there is no
  target column and the artifact is not a scikit-learn `Pipeline`.
- What it answers: "which movies should I watch if I liked this one?", using
  the ratings other people gave.
- Voice commands: "JarvisTEC, recommend a movie", "JarvisTEC, recommend me
  a movie", "JarvisTEC, movie recommendation", "JarvisTEC, what movie can I
  watch", "JarvisTEC, what movie should I watch", "JarvisTEC, suggest a
  movie", "JarvisTEC, I want to watch a movie". To name the movie, add
  "like" or "similar to" and the title: "JarvisTEC, recommend a movie like
  Jurassic Park". Without a title, *Toy Story* is used.

## Dataset

- Source: the movie dataset distributed by the course
  ([Google Drive file](https://drive.google.com/file/d/1Dn1BZD3YxgBQJSIjbfNnmCFlDW2jdQGD)),
  a MovieLens "small"-style extract. MovieLens data is collected and
  published by GroupLens (Harper & Konstan, 2015).
- License: MovieLens datasets are distributed by GroupLens for research and
  education under their own terms of use (citation required, no
  redistribution without permission). The exact terms of this copy were not
  verified; check them before redistributing.
- Expected files: `data/movies/movies.csv` (`movieId`, `title`, `genres`)
  and `data/movies/ratings.csv` (`userId`, `movieId`, `rating`,
  `timestamp`).
- 10,329 movies and 105,339 ratings (0.5 to 5 stars) from 668 users, given
  between April 1996 and January 2016; no missing values, no duplicated
  (user, movie) pairs.
- Very sparse: 1.53 % of the user x movie matrix is filled; the median movie
  has 3 ratings and 35.2 % have only one. 3,855 movies have 5 or more.
- No target: the signal is the rating matrix. A held-out rating of 4 or more
  counts as a relevant movie.

## Configuration

- `SEED = 42`. Random hold-out **per user**: 20 % of each user's ratings is
  the test set (20,812 ratings); 20 % of the rest is a validation set used
  to choose parameters (16,659); 67,868 ratings train.
- Model: cosine similarity between the rating columns of two movies (unrated
  = 0, no mean-centring), multiplied by `n / (n + 100)` where `n` is the
  number of users who rated both; 0 below 3 common users; only movies with
  at least 5 ratings. The table is refitted on all the ratings for export.
- Parameters chosen on validation by NDCG@10 with the "one seed movie"
  protocol (the served scenario): centring {no, yes} x shrink {0, 10, 25,
  50, 100, 200}; smallest shrink within 0.002 NDCG of the best.
- Compared with: random, most rated movies, Bayesian-average rating,
  content-based on genres (TF-IDF + cosine), truncated SVD (30 components).
- Artifact: dict with `movies` (catalogue and rating statistics), `neighbors`
  (20 most similar movies for each of 3,855 movies), `title_index` (22,660
  normalised title keys) and `popular`. Movies with fewer than 5 ratings
  fall back on shared genres ordered by Bayesian-average rating.
- Built with numpy and scikit-learn 1.9.1 (`TruncatedSVD` is only used in
  the comparison; the artifact holds plain Python objects).

## Reproduce

1. Download the two CSV files to `data/movies/`.
2. Open `training/movies/notebook.ipynb` with the `.venv` kernel and run
   every cell (a few minutes; it builds dense user x movie matrices, so
   close memory-hungry programs first).
3. It generates `artifacts/movies.joblib` (0.9 MB, committed) and
   `artifacts/movies.meta.json`.

`normalize`, `canonical` and `title_keys` exist twice: in the export cell of
the notebook (builds the title index) and in
`backend/models/movies/titles.py` (normalises the request). Keep them
identical; `tests/models/test_movies.py` checks that they agree.

## Evaluation (test ratings, 655 users, top-10 lists)

Served scenario: the recommender only knows **one** movie the user liked
(5 random liked movies per user, averaged). `test_metrics` in `meta.json`.

| Approach | HR@10 | P@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|---|
| Popularity (ignores the movie) | 0.5603 | 0.1224 | 0.1120 | **0.1651** | 0.0061 |
| **Item-based CF (served)** | 0.5066 | 0.1004 | 0.1068 | 0.1395 | 0.2012 |
| Truncated SVD factors | 0.3704 | 0.0620 | 0.0712 | 0.0857 | 0.2903 |
| Content-based (genres) | 0.0498 | 0.0053 | 0.0039 | 0.0064 | 0.7565 |

Full history: the recommender sees all the training ratings of the user.
`test_metrics_user_history` in `meta.json`. The item-based row uses a
different configuration from the served table (shrink 10, 200 neighbours,
liked-movies profile, chosen on validation for this protocol); the served
configuration (shrink 100) was not scored with the full history.

| Approach | HR@10 | P@10 | Recall@10 | NDCG@10 | Coverage |
|---|---|---|---|---|---|
| Truncated SVD | 0.7527 | 0.1986 | 0.2057 | **0.2759** | 0.0448 |
| **Item-based CF** | 0.7206 | 0.1979 | 0.1890 | 0.2691 | 0.0435 |
| Popularity (most rated) | 0.5603 | 0.1224 | 0.1120 | 0.1651 | 0.0061 |
| Bayesian average | 0.5557 | 0.1153 | 0.1050 | 0.1433 | 0.0048 |
| Content-based (genres) | 0.0626 | 0.0064 | 0.0035 | 0.0064 | 0.1771 |
| Random | 0.0153 | 0.0015 | 0.0004 | 0.0012 | 0.4714 |

- With the full history, item-based CF beats popularity by 0.104 NDCG
  (bootstrap 95 % interval 0.089 to 0.119) and is tied with SVD (difference
  0.007, interval -0.005 to 0.019).
- **With a single movie it does not beat the popularity baseline** (-0.026
  NDCG, interval -0.038 to -0.014). It is the best approach that depends on
  the movie, and its lists reach 20 % of the catalogue instead of the same
  63 titles for everybody.
- Popularity bias: 76 % of its full-history recommendations are among the
  100 most rated movies. Neighbours of very popular movies are generic
  (*Toy Story* gets *Star Wars* and *Jurassic Park*, and no animated film
  among its first eight neighbours).
- Limitations: cold start (62.7 % of the catalogue has fewer than 5 ratings
  and uses the genre fallback, which was not evaluated on its own), ratings
  end in January 2016, only the MovieLens titles of the data file are
  recognised (an unknown title gets an error with the closest titles), 668
  users. The evaluated lists exclude the movies each user already rated,
  which the API cannot do, and the exported table (refitted on all the
  ratings, 20 neighbours per movie) was not scored itself.

## References

- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based
  collaborative filtering recommendation algorithms. In *Proceedings of the
  10th International Conference on World Wide Web (WWW '01)* (pp. 285-295).
  ACM. https://doi.org/10.1145/371920.372071
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History
  and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4),
  Article 19. https://doi.org/10.1145/2827872
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization
  techniques for recommender systems. *Computer, 42*(8), 30-37.
  https://doi.org/10.1109/MC.2009.263
- Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of recommender
  algorithms on top-N recommendation tasks. In *Proceedings of the Fourth
  ACM Conference on Recommender Systems (RecSys '10)* (pp. 39-46). ACM.
  https://doi.org/10.1145/1864708.1864721
