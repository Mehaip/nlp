"""Shared experiment settings for both authorship-attribution methods."""

CSV_PATH = "Gungor_2018_VictorianAuthorAttribution_data-train.csv"
TEXT_COL = "text"
AUTHOR_COL = "author"

DEV_MAX_AUTHORS = 10
DEV_MAX_SAMPLES_PER_AUTHOR = 100

FINAL_MAX_AUTHORS = 25
FINAL_MAX_SAMPLES_PER_AUTHOR = 500

TEST_SIZE = 0.2
SEED = 42
