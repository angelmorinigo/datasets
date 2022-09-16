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

"""Tests for c4_wsrs_utils."""

from unittest import mock
from absl.testing import absltest
import numpy as np
from tensorflow_datasets.text.c4_wsrs import c4_wsrs_utils


class C4WsrsUtilsTest(absltest.TestCase):

  def test_snippet_contains_word_at_index(self):
    snippet = "the -patient's family is in the emergency room."
    for index, word in [(0, 'the'), (5, 'patient'), (15, 'family'), (22, 'is'),
                        (25, 'in'), (28, 'the'), (32, 'emergency'),
                        (42, 'room')]:
      assert c4_wsrs_utils._snippet_contains_word_at_index(
          snippet, word, index), (word, index)
    assert not c4_wsrs_utils._snippet_contains_word_at_index(snippet, 's', 13)
    assert not c4_wsrs_utils._snippet_contains_word_at_index(snippet, 'fam', 15)

  def test_extract_snippets(self):
    doc = {
        'url': b'test/url.com',
        'text': b'this is a snippet. it has two parts.'
    }
    with mock.patch.object(
        np.random, 'uniform', autospec=True) as random_uniform_mock:
      random_uniform_mock.side_effect = [0.1]  # Ensures periods are added.
      results = list(
          c4_wsrs_utils.extract_snippets(doc, max_sentences_per_snippet=1))
      random_uniform_mock.assert_called_once()
    self.assertEqual(results,
                     [('url=test/url.com,snippet_id=0', 'this is a snippet.'),
                      ('url=test/url.com,snippet_id=1', 'it has two parts.')])

  def test_extract_abbreviation_expansion_pairs(self):
    snippet = 'the patient is in the emergency room.'
    abbreviation_expansions_dict = {'pt': ['patient'], 'er': ['emergency room']}
    result = c4_wsrs_utils._extract_abbreviation_expansion_pairs(
        snippet, abbreviation_expansions_dict)
    self.assertEqual(result, {
        4: [('pt', 'patient')],
        22: [('er', 'emergency room')]
    })

  def test_abbreviate_snippet(self):
    snippet = 'the patient is in the emergency room.'
    index_to_pairs = {4: [('pt', 'patient')], 22: [('er', 'emergency room')]}
    result = c4_wsrs_utils._abbreviate_snippet(
        snippet, index_to_pairs, abbreviation_rate=1.0)
    self.assertEqual(result,
                     ('the pt is in the er.', [('pt', 'patient'),
                                               ('er', 'emergency room')]))

  def test_reverse_substitution(self):
    extracted_snippet = ('url=test/url.com,snippet_id=0',
                         'the patient is in the emergency room.')
    abbreviation_expansions_dict = {'pt': ['patient'], 'er': ['emergency room']}
    rs_results = list(
        c4_wsrs_utils.reverse_substitution(
            extracted_snippet=extracted_snippet,
            abbreviation_rate=1.0,
            abbreviation_expansions_dict=abbreviation_expansions_dict))
    expected_features = c4_wsrs_utils.WSRSFeatures(
        original_snippet='the patient is in the emergency room.',
        abbreviated_snippet='the pt is in the er.')
    self.assertEqual(rs_results,
                     [(('pt', 'patient'),
                       ('url=test/url.com,snippet_id=0', expected_features)),
                      (('er', 'emergency room'),
                       ('url=test/url.com,snippet_id=0', expected_features))])


if __name__ == '__main__':
  absltest.main()
