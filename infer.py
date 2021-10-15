from v1 import *
from pprint import pprint

if __name__ == "__main__":
    v1 = V1()
    v1.load("test-model")
    pprint(
        v1.score(
            "Apple's sales dropped 10 percent today",
            "the most amazing product",
            "your mom is the best",
            "i love your mom",
            "you're the worst",
        )
    )
