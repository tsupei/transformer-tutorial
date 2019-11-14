# Intro.
1. Review transformers both in in theory and in practice. 
2. Run several experiments using traditional-chinese corpus

# Dataset

I crawl some articles from PTT-Gossiping [here](https://drive.google.com/file/d/1XYhk-Nu6DcfGf4DuEeZHPOe6bn7H3BKu/view?usp=sharing)

# Structure

.
├── README.md
├── data
│   ├── Gossiping-38650-39150.json
│   └── transformer.pkl
├── examples
│   └── pytorch_example.py
└── pytorch
    ├── positional_encoding.py
    └── transformer.py

- `data` directory restore data and `transformer.pkl` is a example of trained model
- `pytorch` directory lists useful models that I modified from [Pytorch](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- `pytorch_example.py` list an example of training a transformer based model


# Run

```bash
python ../examples/pytorch_example.py \
--epoch 1 \
--encoder_layer 3 \
--word_dimension 256 \
--hidden_dimension 300 \
--attention_head 8 \
--dropout 0.2 \
--learning_rate 0.05 \
--data /Users/admin/Practice/transformer-tutorial/data/Gossiping-38650-39150.json \
--model /Users/admin/Practice/transformer-tutorial/data/transformer.pkl
```
`TensorboardX` is also supported

```bash
python ../examples/pytorch_example.py \
...\
--tensorboard

```

