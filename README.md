# sentient-sentiment

sentient-setiment is an algorithm developed to extract sentiment values from sentences. It relies heavily on stacked bidirectional and unidirectional LSTM layers to process context clues.
## Docker Setup

Assume you are in the root directory of the project.

### Build the container
```bash
docker build docker -t sentient:latest
```

### Run the container
```bash
docker run -d -it --rm --gpus all -v $HOME:$HOME -w `pwd` -p 8888:8888 --env-file .env --name sentient sentient:latest /bin/bash
```

## Training Dataset

Currently, sentient-sentiment uses a combination of amazon review and financial headline data to train its model. Inputs to the model must adhere to the following conventions:

- Sentiment value column will be indexed `sentiment`
- Text column will be indexed `test`
- Sentiment value column will range between 0 and 1, with 0 meaning negative sentiment and 1 meaning positive sentiment.
  
Alternatively, the variables `TEXT_COL` and `SCORE_COL` defined in `v1.py` can be changed from `text` and `sentiment` respectively.

```
        sentiment                                        text
0             0.0                             A Pain to Clean
1             0.0                             not 1/2" hooks!
2             0.0                                        WEAK
3             0.0                     Not for use as a vacuum
4             0.0            Complete dissasembly required...
...           ...                                         ...
152907        1.0            Tons of beads for a cheap price.
152908        1.0                simple and does the job well
152909        0.0                 I'll be sticking to Leviton
152910        1.0                                  Excellent!
152911        1.0  Great additional to your CrossFit gym bag.
```

## Training

Training is done by providing the model with the data, and calling the `fit` method. `fit` takes a DataFrame in the format of above.

```python
v1 = V1()
data = equalize_distribution(load_data())
v1.fit(data)
v1.save("test-model")
```

Once `fit` is called and the model is trained, `save` can be called. This will save the model and artifacts into the specified directory with structure given below.

```bash
└── test-model
    ├── max_len
    ├── model
    │   ├── assets
    │   ├── keras_metadata.pb
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    └── tokenizer
```

## Inference

To perform inference, a saved model can be loaded via the `load` method. This takes the path to the model directory as the sole parameter. Then, the `score` method can be called to score passed sentences.

```python
v1 = V1()
v1.load("test-model")
scores = v1.score("Apple's sales dropped 15% today")
```