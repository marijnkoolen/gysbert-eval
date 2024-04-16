import os
import random
from typing import Iterable

from gysbert_eval.train_flair_tagger import prep_training
from gysbert_eval.train_flair_tagger import EmbeddingsConfig, TransformerModelConfig, TaggerConfig
from gysbert_eval.train_lm import get_train_test_validate_filenames


def train_layer(trainer, layer_name: str, base_dir: str, learning_rate: float = 0.05,
                mini_batch_size: int = 32, max_epochs: int = 10):
    model_dir = os.path.join(base_dir, f"tagger-layer_{layer_name}")
    results = trainer.train(model_dir,
                            learning_rate=learning_rate,
                            mini_batch_size=mini_batch_size,
                            max_epochs=max_epochs)
    return results


def main():
    # The directory with the train, validate and test data
    layer_name = 'PER'
    corpus_dir = f'ground_truth/tag_de_besluiten/layer_{layer_name}'
    tagger_base_dir = 'resources/taggers/'

    # A binary of a FastText model
    fasttext_file = ('resources/embeddings/fasttext/'
                     'fasttext-dim_384-window_10-min_count_100-case_lower.bin')

    # Configuration of the tagger. These are the properties I've experimented with.
    tagger_config = TaggerConfig(layer_name=layer_name,
                                 use_crf=True, use_rnn=True,
                                 reproject_embeddings=True, hidden_size=256)

    # Configuration of any BERT-based models, in this case GysBERT version 1
    bert_model_config = TransformerModelConfig('emanjavacas/GysBERT',
                                               use_context=True, use_finetuning=False,
                                               layers='-1', subtoken_pooling='first',
                                               allow_long_sentences=False, model_max_length=512)

    # Configuration of all the embeddings that are to be stacked by Flair for training the tagger
    embeddings_configs = [
        EmbeddingsConfig(embed_type='char', embed_file='resources/lm/char_fw/best-lm.pt'),
        EmbeddingsConfig(embed_type='char', embed_file='resources/lm/char_bw/best-lm.pt'),
        EmbeddingsConfig(embed_type='fasttext',
                         embed_file=fasttext_file),
        EmbeddingsConfig(embed_type='bert', model_config=bert_model_config)
    ]

    # Create the Flair trainer (just a wrapper around Huggingface I think)
    trainer = prep_training(tagger_config, embeddings_configs, corpus_dir=corpus_dir)
    train_layer(trainer, layer_name, tagger_base_dir, learning_rate=0.05,
                mini_batch_size=32, max_epochs=10)

