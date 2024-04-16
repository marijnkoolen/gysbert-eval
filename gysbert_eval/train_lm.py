# Modified from https://huggingface.co/blog/how-to-train
import collections
import glob
import logging
import os
import random
from pathlib import Path
from typing import Iterable

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def make_plain_text_train_test_split(corpus_dir: str, texts: Iterable,
                                     test_frac: float = 0.05, validate_frac: float = 0.05) -> None:
    """Create a random train/validate/test split for training a BERT-based model from scratch.

    The code is modified from https://huggingface.co/blog/how-to-train.

    :param corpus_dir: an output directory for writing the train/test/validate text files
    :type corpus_dir: str
    :param texts: an iterable yielding plain texts as items
    :type texts: iterable
    :param test_frac: (optional) fraction of texts to use for test file (default is 0.05)
    :type test_frac: float
    :param validate_frac: (optional) fraction of texts to use for validate file (default is 0.05)
    :type validate_frac: float
    """
    train_dir, test_file, validate_file = get_train_test_validate_filenames(corpus_dir)
    if os.path.exists(train_dir) is False:
        os.mkdir(train_dir)
    train_split = 0
    train_count = 0
    test_count, validate_count = 0, 0
    train_file = f'{corpus_dir}/dummy'
    fh_train = open(train_file, 'wt', encoding="utf-8")
    fh_test = open(test_file, 'wt', encoding="utf-8")
    fh_valid = open(validate_file, 'wt', encoding="utf-8")
    texts_per_split = 1000000

    validate_frac += test_frac

    for text in texts:
        draw = random.random()
        if draw < test_frac:
            fh_test.write(text + '\n')
            test_count += 1
        elif draw < validate_frac:
            fh_valid.write(text + '\n')
            validate_count += 1
        else:
            if train_count % texts_per_split == 0:
                train_split += 1
                train_file = f'{train_dir}/train_split_{train_split}'
                fh_train.close()
                fh_train = open(train_file, 'wt', encoding="utf-8")
            fh_train.write(text + '\n')
            train_count += 1
    print(f'texts - train: {train_count}\tvalidate: {validate_count}\ttest: {test_count}')

    fh_test.close()
    fh_valid.close()
    fh_train.close()


def make_bert_trainer(model_dir: str, text_file: str,
                      tokenizer_max_len: int = 512, mlm_probability: float = 0.15,
                      num_train_epochs: int = 10, per_device_mini_batch_size: int = 64):
    """Create a Trainer object to train a BERT-based model using a plain text file.

    :param model_dir: directory to store the BERT-based model
    :type model_dir: str
    :param text_file: a plain text file with one text (e.g. sentence or paragraph) per line.
    :type text_file: str
    :param tokenizer_max_len: the maximum length of text chunks
    :type tokenizer_max_len: int
    :param mlm_probability: the masked LM probability
    :type mlm_probability: float
    :param num_train_epochs: the number of training epochs
    :type num_train_epochs: int
    :param per_device_mini_batch_size: the mini batch size
    :type per_device_mini_batch_size: int
    """
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, max_len=tokenizer_max_len)
    dataset = load_data_set(text_file, tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_mini_batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True
    )
    model = init_roberta_model()
    return Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )


def load_data_set(file_path: str, tokenizer: Tokenizer):
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )


def init_roberta_model():
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    return RobertaForMaskedLM(config=config)


def load_tokenizer(model_dir: str):
    """Load the trained tokenizer."""
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(model_dir, "vocab.json"),
        os.path.join(model_dir, "merges.txt"),
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    return tokenizer


def train_tokenizer(text_file: str, model_dir: str):
    """Create a tokenizer based on a plain text file."""
    paths = [str(Path(text_file))]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=5, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save_model(model_dir)


def get_train_test_validate_filenames(corpus_dir: str):
    train_dir = f"{corpus_dir}/train"
    test_file = f"{corpus_dir}/test.txt"
    validate_file = f"{corpus_dir}/valid.txt"
    return train_dir, test_file, validate_file


def make_character_dictionary(corpus_dir: str):
    train_dir, test_file, validate_file = get_train_test_validate_filenames(corpus_dir)

    char_dictionary: Dictionary = Dictionary()

    # counter object
    char_freq = collections.Counter()

    files = glob.glob(f'{train_dir}/*')
    files += [test_file, validate_file]

    logging.info('making character dictionary')
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                chars = list(line)
                char_freq.update(chars)

    total_count = sum(char_freq.values())

    logging.info(f'\ttotal character count: {total_count}')

    cumu = 0
    idx = 0
    for letter, count in char_freq.most_common():
        cumu += count
        percentile = (cumu / total_count)

        # comment this line in to use only top X percentile of chars, otherwise filter later
        # if percentile < 0.00001: break

        char_dictionary.add_item(letter)
        idx += 1
        logging.info('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, cumu, percentile))

    import pickle
    with open(f'{corpus_dir}/char_mappings', 'wb') as f:
        mappings = {
            'idx2item': char_dictionary.idx2item,
            'item2idx': char_dictionary.item2idx
        }
        pickle.dump(mappings, f)


def train_lm(corpus_dir: str, is_forward_lm: bool = True, character_level: bool = True,
             hidden_size: int = 128, nlayers: int = 1, sequence_length: int = 250,
             mini_batch_size: int = 100, max_epochs: int = 10):
    # are you training a forward or backward LM?

    # load the default character dictionary
    dictionary = Dictionary.load_from_file(f'{corpus_dir}/char_mappings')

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(corpus_dir,
                        dictionary,
                        is_forward_lm,
                        character_level=character_level)

    # instantiate your language model, set hidden size and number of layers
    language_model = LanguageModel(dictionary,
                                   is_forward_lm,
                                   hidden_size=hidden_size,
                                   nlayers=nlayers)

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus)

    level = 'char' if character_level else 'word'
    direction = 'fw' if is_forward_lm else 'bw'

    trainer.train(f'{corpus_dir}/resources/taggers/language_model_{level}_{direction}',
                  sequence_length=sequence_length,
                  mini_batch_size=mini_batch_size,
                  max_epochs=max_epochs)
    return None
