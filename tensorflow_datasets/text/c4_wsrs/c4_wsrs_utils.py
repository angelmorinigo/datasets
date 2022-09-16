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

"""Utils to process C4-WSRS."""

from __future__ import annotations
import dataclasses
import random
from typing import Iterator, Mapping, MutableSequence, Sequence
import numpy as np


@dataclasses.dataclass
class WSRSFeatures:
  """Stores the original snippet and the snippet abbreviated by WSRS."""
  original_snippet: str = ''
  abbreviated_snippet: str = ''


def _get_dictionary_pairs(dictionary: Mapping[str, Sequence[str]]):
  for abbreviation, expansions in dictionary.items():
    for expansion in expansions:
      yield abbreviation, expansion


def _snippet_contains_word_at_index(snippet: str, word: str,
                                    index: int) -> bool:
  """Checks if the snippet contains the word at the given index."""
  if snippet[index:index + len(word)] != word:
    return False
  # Ensures the word is not a substring of a larger word.
  if index > 0 and snippet[index - 1].isalnum():
    return False
  if index > 1 and snippet[index - 1] == "'" and snippet[index - 2].isalnum():
    return False
  if ((index + len(word)) < len(snippet) and
      snippet[index + len(word)].isalnum()):
    return False
  return True


def extract_snippets(
    c4_doc: Mapping[str, bytes],
    max_sentences_per_snippet: int) -> Iterator[tuple[str, str]]:
  """Extracts variable-length multi-sentence snippets from C4 web text."""
  text = c4_doc['text'].decode()
  text = text.replace('\n', ' ')
  sentences = text.split('. ')
  sentence_idx = 0
  snippet_id = 0
  while sentence_idx < len(sentences):
    num_sentences_sampled = np.random.randint(1, max_sentences_per_snippet + 1)
    snippet = ('. '.join(sentences[sentence_idx:sentence_idx +
                                   num_sentences_sampled]))
    if not snippet.endswith('.') and np.random.uniform() < 0.5:
      snippet += '.'
    key = f'url={c4_doc["url"].decode()},snippet_id={snippet_id}'
    snippet_id += 1
    sentence_idx += num_sentences_sampled
    snippet = snippet.lower().strip()
    if len(snippet) > 1024:
      continue
    yield key, snippet


def _extract_abbreviation_expansion_pairs(
    snippet: str, abbreviation_expansions_dict: Mapping[str, Sequence[str]]
) -> dict[int, list[tuple[str, str]]]:
  """Extracts all possible abbreviation-expansion pairs from a snippet."""
  snippet_idx = 0
  index_to_pairs = {}
  while snippet_idx < len(snippet):
    pairs = []
    for abbreviation, expansion in _get_dictionary_pairs(
        abbreviation_expansions_dict):
      if not _snippet_contains_word_at_index(snippet, expansion, snippet_idx):
        continue
      pairs.append((abbreviation, expansion))
    if pairs:
      index_to_pairs[snippet_idx] = pairs
    snippet_idx += 1
  return index_to_pairs


def _abbreviate_snippet(
    snippet: str, index_to_pairs: Mapping[int, MutableSequence[tuple[str,
                                                                     str]]],
    abbreviation_rate: float) -> tuple[str, list[tuple[str, str]]]:
  """Applies abbreviations to a snippet."""
  snippet_idx = 0
  abbreviated_text = ''
  replacements = []
  while snippet_idx < len(snippet):
    if snippet_idx in index_to_pairs:
      pairs_and_counts = index_to_pairs[snippet_idx]
      random.shuffle(pairs_and_counts)
      abbreviation, expansion = pairs_and_counts[0]
      if np.random.uniform(0, 1) < abbreviation_rate:
        replacements.append((abbreviation, expansion))
        abbreviated_text += abbreviation
        snippet_idx += len(expansion)
        continue
    abbreviated_text += snippet[snippet_idx]
    snippet_idx += 1
  return abbreviated_text, replacements


def reverse_substitution(
    extracted_snippet: tuple[str, str], abbreviation_rate: float,
    abbreviation_expansions_dict: Mapping[str, Sequence[str]]
) -> Iterator[tuple[tuple[str, str], tuple[str, WSRSFeatures]]]:
  """Conducts reverse substitution.

  Yields one WSRSFeatures obj per replacement.

  Args:
    extracted_snippet: The snippet extracted from the c4 doc, with the url as
      key.
    abbreviation_rate: The rate at which expansions are abbreviated.
    abbreviation_expansions_dict: A dictionary mapping each abbreviation to its
      valid expansions.

  Yields:
    A key-value tuple in which the key is a tuple containing the
    abbreviation-expansion pair replacement, and the value is a tuple containing
    the original key and a WSRSFeatures object holding the extracted snippet and
    its abbreviated form.
  """

  key, snippet = extracted_snippet
  snippet_idx_to_pairs = (
      _extract_abbreviation_expansion_pairs(snippet,
                                            abbreviation_expansions_dict))
  abbreviated_snippet, replacements = snippet, []
  if snippet_idx_to_pairs:
    abbreviated_snippet, replacements = _abbreviate_snippet(
        snippet, snippet_idx_to_pairs, abbreviation_rate)
  if len(abbreviated_snippet.split(' ')) < 3:
    return
  features = WSRSFeatures()
  features.original_snippet = snippet
  features.abbreviated_snippet = abbreviated_snippet
  if not replacements:
    yield ('', ''), (key, features)
  for replacement in replacements:
    yield replacement, (key, features)


def sample_snippets_by_replacement(
    snippets_by_replacement: tuple[tuple[str, str],
                                   Sequence[tuple[str, WSRSFeatures]]],
    num_snippets_per_replacement: int) -> Iterator[tuple[str, WSRSFeatures]]:
  """Samples a fixed number of snippets for each replacement pair."""
  limit = num_snippets_per_replacement
  if limit == 0:
    return
  snippets = snippets_by_replacement[1]
  for key, features in snippets:
    yield key, features
    limit -= 1
    if limit <= 0:
      break
