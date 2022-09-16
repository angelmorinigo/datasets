# coding=utf-8
# Copyright 2022 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""C4-WSRS dataset."""

from __future__ import annotations
import collections
import csv
from typing import Mapping, Sequence
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.text.c4_wsrs import c4_wsrs_utils

WSRSFeatures = c4_wsrs_utils.WSRSFeatures

_ABBREV_EXPANSION_DICT_URI = 'gs://gresearch/deciphering_clinical_abbreviations/abbreviation_expansion_dictionary.csv'

_DESCRIPTION = """\
A medical abbreviation expansion dataset which applies web-scale reverse
substitution (wsrs) to the C4 dataset, which is a colossal, cleaned version of
Common Crawl's web crawl corpus.

The original source is the Common Crawl dataset: https://commoncrawl.org
"""
_CITATION = """
@article{rajkomar2022decipherclinicalabbrev,
 author  = {Alvin Rajkomar and Eric Loreaux and Yuchen Liu and Jonas Kemp and Benny Li and Ming-Jun Chen and Yi Zhang and Afroz Mohiuddin and Juraj Gottweis},
 title   = {Deciphering clinical abbreviations with a privacy protecting machine learning system},
 journal = {Nature Communications},
 year    = {2022},
}
"""


def _convert_abbrev_expansion_table_to_dict(
    df: pd.DataFrame) -> dict[str, list[str]]:
  abbreviation_expansions = {}
  for row_tuple in df.itertuples(index=False):
    abbreviation, expansion = row_tuple.abbreviation, row_tuple.expansion
    if abbreviation not in abbreviation_expansions:
      abbreviation_expansions[abbreviation] = []
    if expansion not in abbreviation_expansions[abbreviation]:
      abbreviation_expansions[abbreviation].append(expansion)
  return abbreviation_expansions


class C4WSRSConfig(tfds.core.BuilderConfig):
  """BuilderConfig for C4-WSRS dataset."""

  def __init__(self, name: str, max_sentences_per_snippet: int,
               abbreviation_rate: float, num_snippets_per_replacement: int,
               **kwargs):
    """Initializes the BuilderConfig for C4-WSRS.

    Args:
      name: The name for the config.
      max_sentences_per_snippet: The maximum number of sentences that can be
        combined into a single snippet. The number of sentences combined into
        each snippet will be a randomly sampled integer from 1 to
        max_sentences_per_snippet.
      abbreviation_rate: The rate at which expansions are abbreviated.
      num_snippets_per_replacement: The number of snippets to sample for each
        unique abbreviation-expansion replacement.
      **kwargs: keyword arguments forwarded to super.
    """
    super().__init__(name=name, **kwargs)
    self.max_sentences_per_snippet = max_sentences_per_snippet
    self.abbreviation_rate = abbreviation_rate
    self.num_snippets_per_replacement = num_snippets_per_replacement


class C4WSRS(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for C4-WSRS dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      C4WSRSConfig(
          'default',
          max_sentences_per_snippet=3,
          abbreviation_rate=0.95,
          num_snippets_per_replacement=4000,
          description='Default C4-WSRS dataset.'),
      C4WSRSConfig(
          'deterministic',
          max_sentences_per_snippet=1,
          abbreviation_rate=1.0,
          num_snippets_per_replacement=4000,
          description='Default C4-WSRS dataset.'),
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = {
        'original_snippet': tfds.features.Text(),
        'abbreviated_snippet': tfds.features.Text(),
    }
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        citation=_CITATION,
        homepage='https://github.com/google-research/google-research/tree/master/medical_abbreviation_expansion',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    abbreviation_expansions_dict_file = dl_manager.download(
        _ABBREV_EXPANSION_DICT_URI)
    abbreviation_expansions_dict = collections.defaultdict(list)
    with tf.io.gfile.GFile(abbreviation_expansions_dict_file) as f:
      reader = csv.reader(f)
      for row in reader:
        abbrev, exp = row
        abbreviation_expansions_dict[abbrev].append(exp)
    return {
        'train': self._generate_examples('train', abbreviation_expansions_dict),
        'test': self._generate_examples('test', abbreviation_expansions_dict),
    }

  def _generate_examples(self, split: str,
                         abbreviation_expansions_dict: Mapping[str,
                                                               Sequence[str]]):
    """Yields examples."""

    def _process_example(element: tuple[str, WSRSFeatures]):
      key, features = element
      return key, {
          'original_snippet': features.original_snippet,
          'abbreviated_snippet': features.abbreviated_snippet,
      }

    beam = tfds.core.lazy_imports.apache_beam

    builder = tfds.builder('c4', config='en', version='3.1.0')
    return (
        tfds.beam.ReadFromTFDS(builder, split=split)
        | 'AsNumpy' >> beam.Map(tfds.as_numpy)
        | 'ExtractSnippets' >> beam.FlatMap(
            c4_wsrs_utils.extract_snippets,
            self.builder_config.max_sentences_per_snippet)
        | 'ReshuffleSnippets1' >> beam.Reshuffle()
        | 'ReverseSubstitution' >> beam.FlatMap(
            c4_wsrs_utils.reverse_substitution,
            self.builder_config.abbreviation_rate, abbreviation_expansions_dict)
        | 'GroupByReplacement' >> beam.GroupByKey()
        | 'SampleSnippetsByReplacement' >> beam.FlatMap(
            c4_wsrs_utils.sample_snippets_by_replacement,
            self.builder_config.num_snippets_per_replacement)
        | 'ReshuffleSnippets2' >> beam.Reshuffle()
        | 'RemoveDuplicates' >> beam.CombinePerKey(lambda vs: next(iter(vs)))
        | 'ProcessExamples' >> beam.Map(_process_example))
