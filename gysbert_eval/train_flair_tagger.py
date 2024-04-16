import os.path
from dataclasses import dataclass
from typing import List

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import StackedEmbeddings, FlairEmbeddings
from flair.embeddings import FastTextEmbeddings
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


@dataclass
class TransformerModelConfig:

    model_name: str
    use_context: bool = False
    use_finetuning: bool = False
    layers: str = "-1"
    subtoken_pooling: str = "first"
    allow_long_sentences: bool = False
    model_max_length: int = 512


@dataclass
class TaggerConfig:

    layer_name: str
    use_crf: bool = True
    use_rnn: bool = True
    reproject_embeddings: bool = False
    hidden_size: int = 256


class EmbeddingsConfig:

    def __init__(self, embed_type: str, embed_dir: str = None,
                 embed_file: str = None, model_config: TransformerModelConfig = None):
        self.embed_type = embed_type
        self.embed_dir = embed_dir
        self.embed_file = embed_file
        self.model_config = model_config


def load_bert_model(model_config: TransformerModelConfig):
    """Load a BERT-based model.

    E.g.
    - 'emanjavacas/GysBERT'
    - 'emanjavacas/GysBERT-v2-1m'
    - 'emanjavacas/GysBERT-v2-1.5m'
    - 'emanjavacas/GysBERT-v2-2m'
    """
    return TransformerWordEmbeddings(model_config.model_name,
                                     layers=model_config.layers,
                                     subtoken_pooling=model_config.subtoken_pooling,
                                     fine_tune=model_config.use_finetuning,
                                     use_context=model_config.use_context,
                                     allow_long_sentences=model_config.allow_long_sentences,
                                     model_max_length=model_config.model_max_length)


def prep_embeddings(embeddings_configs: List[EmbeddingsConfig]) -> StackedEmbeddings:
    """Load all embeddings to be stacked for training."""
    embeddings_stack = []
    for embed_conf in embeddings_configs:
        if embed_conf.embed_type == 'character_lm':
            char_lm = FlairEmbeddings(embed_conf.embed_file)
            embeddings_stack.append(char_lm)

        elif embed_conf.embed_type == 'fasttext':
            fasttext_embeddings = FastTextEmbeddings(embed_conf.embed_file)
            embeddings_stack.append(fasttext_embeddings)

        elif embed_conf.embed_type == 'bert':
            bert_model = load_bert_model(embed_conf.model_config)
            embeddings_stack.append(bert_model)

    if len(embeddings_stack) == 0:
        return None
    return StackedEmbeddings(embeddings=embeddings_stack)


def prep_corpus(data_dir: str, layer_name: str = 'ner') -> ColumnCorpus:
    """Create a corpus object based on column-formatted (csv/tsv) ground truth data.

    The data directory has to contain three files:
    - train.txt
    - test.txt
    - validate.txt

    Each file needs to be in column format, e.g.:

    <token> <tag>

    E.g.:
    I O
    am O
    Gysbert B-PER
    """
    train_file = os.path.join(data_dir, f'train.txt')
    test_file = os.path.join(data_dir, 'test.txt')
    validate_file = os.path.join(data_dir, 'validate.txt')
    assert os.path.exists(train_file), f"the train file {train_file} doesn't exist"
    assert os.path.exists(test_file), f"the test file {test_file} doesn't exist"
    assert os.path.exists(validate_file), f"the validate file {validate_file} doesn't exist"

    columns = {0: 'text', 1: layer_name}

    return ColumnCorpus(data_dir, columns,
                        train_file=f'train.txt',
                        test_file='test.txt',
                        dev_file='validate.txt')


def prep_trainer(tagger_config: TaggerConfig, corpus: Corpus,
                 embeddings: StackedEmbeddings,
                 label_type: str = 'ner') -> ModelTrainer:
    """Prepare a trainer model for a sequence tagger."""

    label_dict = corpus.make_label_dictionary(label_type=label_type)
    tagger = SequenceTagger(hidden_size=tagger_config.hidden_size,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=tagger_config.use_crf,
                            use_rnn=tagger_config.use_rnn,
                            reproject_embeddings=tagger_config.reproject_embeddings)

    return ModelTrainer(tagger, corpus)


def prep_training(tagger_config: TaggerConfig,
                  embeddings_configs: List[EmbeddingsConfig],
                  corpus_dir: str) -> ModelTrainer:
    corpus = prep_corpus(corpus_dir, tagger_config.layer_name)

    embeddings = prep_embeddings(embeddings_configs)

    if embeddings is None:
        raise ValueError('no embeddings configuration passed')
    return prep_trainer(tagger_config, corpus, embeddings)

