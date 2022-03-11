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

"""SAVEE dataset."""

import collections
import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

LABEL_MAP = {
    'angry': 'anger',
    'sad': 'sadness',
    'fear': 'fear',
    'neutral': 'neutral',
    'happy': 'happiness'
}


_SAMPLE_RATE = 24414


_CITATION = """
@data{SP2/E8H2MF_2020,
author = {Pichora-Fuller, M. Kathleen and Dupuis, Kate},
publisher = {Scholars Portal Dataverse},
title = {{Toronto emotional speech set (TESS)}},
year = {2020},
version = {DRAFT VERSION},
doi = {10.5683/SP2/E8H2MF},
url = {https://doi.org/10.5683/SP2/E8H2MF}
}
"""

_DESCRIPTION = """
These stimuli were modeled on the Northwestern University Auditory Test No. 6
(NU-6; Tillman & Carhart, 1966). A set of 200 target words were spoken in
the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years)
and recordings were made of the set portraying each of seven emotions (anger,
disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are
2800 stimuli in total. Two actresses were recruited from the Toronto area.
Both actresses speak English as their first language, are university educated,
and have musical training. Audiometric testing indicated that both actresses have
thresholds within the normal range. (2010-06-21)
"""

_HOMEPAGE = "https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi%3A10.5683%2FSP2%2FE8H2MF"


def parse_name(name, from_i, to_i, mapping=None):
    """Source: https://audeering.github.io/audformat/emodb-example.html"""
    key = name[from_i:to_i]
    return mapping[key] if mapping else key


def _compute_split_boundaries(split_probs, n_items):
  """Computes boundary indices for each of the splits in split_probs.

  Args:
    split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
      ('test', 0.2)]
    n_items: Number of items we want to split.

  Returns:
    The item indices of boundaries between different splits. For the above
    example and n_items=100, these will be
    [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)].
  """
  if len(split_probs) > n_items:
    raise ValueError('Not enough items for the splits. There are {splits} '
                     'splits while there are only {items} items'.format(splits=len(split_probs), items=n_items))
  total_probs = sum(p for name, p in split_probs)
  if abs(1 - total_probs) > 1E-8:
    raise ValueError('Probs should sum up to 1. probs={}'.format(split_probs))
  split_boundaries = []
  sum_p = 0.0
  for name, p in split_probs:
    prev = sum_p
    sum_p += p
    split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

  # Guard against rounding errors.
  split_boundaries[-1] = (split_boundaries[-1][0], split_boundaries[-1][1],
                          n_items)

  return split_boundaries


def _get_inter_splits_by_group(items_and_groups, split_probs, split_number):
  """Split items to train/dev/test, so all items in group go into same split.

  Each group contains all the samples from the same speaker ID. The samples are
  splitted so that all each speaker belongs to exactly one split.

  Args:
    items_and_groups: Sequence of (item_id, group_id) pairs.
    split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
      ('test', 0.2)]
    split_number: Generated splits should change with split_number.

  Returns:
    Dictionary that looks like {split name -> set(ids)}.
  """
  groups = sorted(set(group_id for item_id, group_id in items_and_groups))
  rng = np.random.RandomState(split_number)
  rng.shuffle(groups)

  split_boundaries = _compute_split_boundaries(split_probs, len(groups))
  group_id_to_split = {}
  for split_name, i_start, i_end in split_boundaries:
    for i in range(i_start, i_end):
      group_id_to_split[groups[i]] = split_name

  split_to_ids = collections.defaultdict(set)
  for item_id, group_id in items_and_groups:
    split = group_id_to_split[group_id]
    split_to_ids[split].add(item_id)

  return split_to_ids


class Tess(tfds.core.GeneratorBasedBuilder):
  """The audio part of TESS dataset for emotion recognition."""

  VERSION = tfds.core.Version('1.0.0')

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain the file dataverse_files.zip.
  """

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'audio': tfds.features.Audio(file_format='wav', sample_rate=_SAMPLE_RATE),
            'label': tfds.features.ClassLabel(names=list(LABEL_MAP.values())),
            'speaker_id': tf.string
        }),
        supervised_keys=('audio', 'label'),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    zip_path = os.path.join(dl_manager.manual_dir, 'dataverse_files.zip')

    if not tf.io.gfile.exists(zip_path):
      raise AssertionError(
          'TESS requires manual download of the data. Please download '
          'the audio data and place it into: {}'.format(zip_path))
    # Need to extract instead of reading directly from archive since reading
    # audio files from zip archive is not supported.
    extract_path = dl_manager.extract(zip_path)

    items_and_groups = []
    for fname in tf.io.gfile.glob('{}/*.wav'.format(extract_path)):
      if os.path.basename(fname) in ["YAF_neat_fear.wav", "YAF_germ_angry.wav"]:
        continue
      if os.path.basename(fname).split("_")[-1][:-4] in ['angry','fear','happy','neutral', 'sad']:
        speaker_id = parse_name(os.path.basename(fname), from_i=0, to_i=3)
        items_and_groups.append((fname, speaker_id))

    split_probs = [('train', 0.8), ('test', 0.2)]
    splits = _get_inter_splits_by_group(items_and_groups, split_probs, 0)

    with open("train.lst", 'w') as f:
        for l in splits['train']:
            f.write(l+"\n")

    with open("test.lst", 'w') as f:
        for l in splits['test']:
            f.write(l+"\n")

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={'file_names': splits['train']},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={'file_names': splits['test']},
        ),
    ]

  def _generate_examples(self, file_names):
    """Yields examples."""
    for fname in file_names:
      wavname = os.path.basename(fname)
      speaker_id = parse_name(wavname, from_i=0, to_i=3)
      label = LABEL_MAP[wavname.split("_")[-1][:-4]]
      example = {'audio': fname, 'label': label, 'speaker_id': speaker_id}
      yield fname, example