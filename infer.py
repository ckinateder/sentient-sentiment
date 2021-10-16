from v1 import *
from pprint import pprint

if __name__ == "__main__":
    v1 = V1()
    v1.load("test-model")
    pprint(
        v1.score(
            "Bitcoin tops $60,000 for first time in six months as traders bet on ETF approval",
            "Bitcoin $100,000 may be conservative, analyst says",
            "Bitcoin Rally Reaches Its Risky Level For October",
            "Dow adds 300 points following earnings beats, surprise retail sales gain",
            "Professor who called Dow 20,000 says he’s nervous about trends in inflation that could spark a stock-market correction",
        )
    )
