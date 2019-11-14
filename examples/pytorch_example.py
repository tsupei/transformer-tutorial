import json
import argparse
import re
from tqdm import tqdm
import torch
import torch.nn as nn
from pytorch.transformer import TransformerModel


def read_json(filename):
    """ Read raw data from json file
    Args:
        filename (str): the path to the file we intend to read

    Raises:
        IOError: Fail to read file
        ValueError: Data is empty

    Returns:
        samples (list): the list of input sentences
    """
    try:
        with open(filename, 'r', encoding="utf8") as file:
            samples = []
            data = json.loads(file.read())
            for article in data["articles"]:
                if "content" not in article:
                    continue
                samples.append(article["content"])
            if not samples:
                raise ValueError("File is empty: {}".format(filename))
            return samples
    except ValueError as err:
        raise err
    except Exception:
        raise IOError("Fail to read file: {}".format(filename))


def read_data(filename):
    """ Read raw data from file
    Args:
        filename (str): the path to the file we intend to read

    Raises:
        IOError: Fail to read file
        ValueError: Data is empty

    Returns:
        samples (list): the list of input sentences
    """
    try:
        with open(filename, 'r', encoding="utf8") as file:
            samples = file.read().split('\n')
            samples = filter(lambda x:x, samples)
            if not samples:
                raise ValueError("File is empty: {}".format(filename))
            return samples
    except ValueError as err:
        raise err
    except Exception:
        raise IOError("Fail to read file: {}".format(filename))


def clean_data(data):
    """
    Args:
        data (str): A sentence

    Returns:
        data (str): A sentence (after cleaning)

    """
    pattern_url = re.compile(r'https?.*?\s')
    pattern_noise = re.compile(r'(蛤|傻眼|痾|呵)+')
    pattern_no_zh_en = re.compile(r'[^\u4e00-\u9fff|^a-zA-Z]+')

    data = re.sub(pattern_url, ' ', data)
    data = re.sub(pattern_noise, ' ', data)
    data = re.sub(pattern_no_zh_en, ' ', data)
    return data


def indexing(data, padding=False):
    """ Turn sentence into index manner
    For examples:
        [...['我是一隻魚']...]
    ==>
        [...[5, 20, 543, 123, 99]...]
    Args:
        data (list): list of sentences (in string manner)

    Returns:
        idata (list): list of sentences (in index manner)
        char_dict (dict): mapping of characters
        vocab_len (int): total amount of character
    """
    # Create charset
    charset = set()
    if padding:
        charset.add("<CLS>")
        charset.add("<SEP>")
    for sentence in data:
        for char in sentence:
            charset.add(char)

    # Create the dict for mapping character to an unique id
    char_dict = dict()
    for idx, char in enumerate(list(charset)):
        char_dict.setdefault(char, idx)

    # Turn sentences into index arrays
    idata = []
    for sentence in data:
        sentence = clean_data(sentence)
        if len(sentence) < 50:
            print(sentence)
            continue
        sentence = sentence[:50]
        index_arr = []
        if padding:
            index_arr.append(char_dict["<CLS>"])
        for char in sentence:
            if char == ' ':
                char = "<SEP>"
            index_arr.append(char_dict[char])
        if padding:
            index_arr.append(char_dict["<SEP>"])
        idata.append(index_arr)
    return idata, char_dict, len(char_dict)


def batchify(data, bsz, drop_last=False):
    """ Chunk data into batch
    Args:
        data (list): list of samples
        bsz (int): batch size
    Returns:
        bdata (list): list of batches
        total (int): The number of batches
    """
    # Count the total amount of batches
    total_batch = len(data) // bsz
    if not drop_last:
        if len(data) % bsz != 0:
            total_batch += 1
    if total_batch == 0:
        raise ValueError("Batch size({bsz}) is bigger than the data size({dsz})".format(bsz=bsz, dsz=len(data)))

    # Chunk into Batch
    bdata = []
    for idx in range(total_batch):
        bdata.append(data[idx * bsz : (idx+1) * bsz])

    return bdata, len(bdata)


def get_batch(batch):
    feature = []
    target = []
    for seq in batch:
        feature.append(seq[:-1])
        target.append(seq[1:])
    return torch.tensor(feature, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def origin_seq(char_dict, seq):
    char_list = [' '] * len(char_dict)
    for key, value in char_dict.items():
        if value >= len(char_dict):
            print('error')
            continue
        char_list[value] = key
    return [char_list[char_idx] for char_idx in seq]


def save_model(model_state, path_to_file):
    torch.save(model_state, path_to_file)


def load_model(path_to_file):
    return torch.load(path_to_file)["state"]


def train(args):
    # Parameters
    path_to_file = args.data
    num_of_epochs = args.epoch
    num_of_head = args.attention_head
    num_of_layer = args.encoder_layer
    dim_word = args.word_dimension
    dim_hidden = args.hidden_dimension
    dropout = args.dropout
    lr = args.learning_rate

    # Read data from file
    samples = read_json(path_to_file)

    # Turn data into index form
    samples, char_dict, vocab_len = indexing(samples, padding=True)

    # Batchify the data
    samples, nbatch = batchify(samples, bsz=8, drop_last=True)

    # Training Settings
    model = TransformerModel(num_of_vocab=vocab_len,
                             num_of_head=num_of_head,
                             num_of_layer=num_of_layer,
                             dim_word=dim_word,
                             dim_hidden=dim_hidden,
                             dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Train
    log_interval = nbatch // 10
    for epoch in range(num_of_epochs):
        with tqdm(total=nbatch) as pg_bar:
            step = 0
            for sample in samples:
                # N(batch-size), S(sequence-size)
                feature, target = get_batch(sample)
                optimizer.zero_grad()
                output = model(feature)
                loss = criterion(input=output.permute(0, 2, 1), target=target)
                if step % log_interval == 0:
                    tqdm.write("Mini-batch: {}/{}".format(step, nbatch))
                    tqdm.write("Loss: {}".format(loss.item()))
                    tqdm.write("Target: {}".format(origin_seq(char_dict, target[0].data)))

                    _, pred = torch.max(output[0], 1)

                    tqdm.write("Output: {}".format(origin_seq(char_dict, pred)))
                loss.backward()
                optimizer.step()
                pg_bar.update(1)
                step += 1
    save_model(model.state_dict(), args.model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", "-e", dest="epoch", type=int, required=True)
    parser.add_argument("--encoder_layer", "-el", dest="encoder_layer", type=int, required=True)
    parser.add_argument("--word_dimension", "-wd", dest="word_dimension", type=int, required=True)
    parser.add_argument("--hidden_dimension", "-hd", dest="hidden_dimension", type=int, required=True)
    parser.add_argument("--attention_head", "-ah", dest="attention_head", type=int, required=True)
    parser.add_argument("--dropout", "-dr", dest="dropout", type=float, required=True)
    parser.add_argument("--leaning_rate", "-lr", dest="learning_rate", type=float, required=True)

    parser.add_argument("--data", dest="data", required=True)
    parser.add_argument("--model", dest="model", required=True)

    args = parser.parse_args()

    train(args)

