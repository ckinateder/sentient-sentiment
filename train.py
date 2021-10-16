from v1 import *
import jsonlines
from tabulate import tabulate


def load_data():
    """load datasets"""

    def read_jsonl(path: str) -> pd.DataFrame:
        out = []
        with jsonlines.open(path) as f:
            for line in f.iter():
                out.append(line)
        return pd.DataFrame(out)

    def filter(df: pd.DataFrame, col: str):
        df[col] = np.select(
            [df[col] > 3, df[col] == 3, df[col] < 3],
            ["positive", "neutral", "negative"],
        )

    # from http://jmcauley.ucsd.edu/data/amazon/
    amazon_set_1 = read_jsonl("datasets/reviews_Musical_Instruments_5.json")[
        ["summary", "overall"]
    ].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_1, SCORE_COL)

    amazon_set_2 = read_jsonl("datasets/reviews_Office_Products_5.json")[
        ["summary", "overall"]
    ].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_2, SCORE_COL)

    amazon_set_3 = read_jsonl("datasets/reviews_Tools_and_Home_Improvement_5.json")[
        ["summary", "overall"]
    ].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_3, SCORE_COL)

    amazon_set_4 = read_jsonl("datasets/reviews_Toys_and_Games_5.json")[
        ["summary", "overall"]
    ].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_4, SCORE_COL)

    amazon_set_5 = read_jsonl("datasets/reviews_Home_and_Kitchen_5.json")[
        ["summary", "overall"]
    ].rename(columns={"summary": TEXT_COL, "overall": SCORE_COL})
    filter(amazon_set_5, SCORE_COL)

    finance_set_1 = pd.read_csv(
        "datasets/financial-headlines.csv", encoding="ISO-8859-1"
    )
    finance_set_1.columns = [SCORE_COL, TEXT_COL]

    data = pd.concat(
        [
            finance_set_1,
            amazon_set_1,
            amazon_set_2,
            amazon_set_3,
            amazon_set_4,
            amazon_set_5,
        ]
    )
    # remove neutral rows cause they do not matter
    data = data[data[SCORE_COL] != "neutral"]
    data = data.sample(frac=1).reset_index(drop=True)  # shuffle

    data[SCORE_COL] = np.select(
        [data[SCORE_COL] == "positive", data[SCORE_COL] == "negative"],
        [1, 0],
    )  # convert to 1 for positive and 0 for negative

    return data


def equalize_distribution(
    data: pd.DataFrame, pos_val: float = 1, neg_val: float = 0
) -> pd.DataFrame:
    """Evenly distribute dataset. Use wisely"""
    pos_count = data[data[SCORE_COL] == pos_val].shape[0]
    neg_count = data[data[SCORE_COL] == neg_val].shape[0]
    if pos_count > neg_count:
        data[data[SCORE_COL] == pos_val] = data[data[SCORE_COL] == pos_val][-neg_count:]
        data = data.dropna()
    elif neg_count > pos_count:
        data[data[SCORE_COL] == neg_val] = data[data[SCORE_COL] == neg_val][-pos_count:]
        data = data.dropna()
    return data


if __name__ == "__main__":
    v1 = V1(epochs=8)
    data = equalize_distribution(load_data())
    v1.fit(data)
    v1.save("test-model")
