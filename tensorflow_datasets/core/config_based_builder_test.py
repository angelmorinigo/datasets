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

"""Tests for config_based_builder."""

from tensorflow_datasets import testing
from tensorflow_datasets.testing.dummy_config_based_datasets.description_citation import builder


class ConfigBasedBuilderTest(testing.TestCase):

  def test_class_named_after_pkg_name(self):
    ds_builder = builder.Builder()
    self.assertEqual(ds_builder.name, "description_citation")

  def test_description_citation_read_from_config(self):
    ds_builder = builder.Builder()
    info = ds_builder._info()
    self.assertEqual(
        info.description,
        "Description of `description_citation` dummy config-based dataset.")
    self.assertEqual(
        info.citation, """@Article{google22tfds,
author = "The TFDS team",
title = "TFDS: a collection of ready-to-use datasets for use with TensorFlow, Jax, and other Machine Learning frameworks.",
journal = "ML gazette",
year = "2022"
}""")


if __name__ == "__main__":
  testing.test_main()
