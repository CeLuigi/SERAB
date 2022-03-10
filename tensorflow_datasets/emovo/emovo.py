"""emovo dataset."""

import collections
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from audio_features import Audio


# Markdown description    that will appear on the catalog page.
_DESCRIPTION = """
An emotional corpus applicable to Italian language.
The database is built from the voices of up to 6 actors who played 14 sentences simulating 6 emotional states (disgust, fear, anger, joy, surprise, sadness)
plus the neutral state.
Each recording session is about an hour.
While most of the original recordings are in stereo, all audios were here converted to mono.
"""

# BibTeX citation
_CITATION = """
@inproceedings{costantini2014emovo,
    title={EMOVO corpus: an Italian emotional speech database},
    author={Costantini, Giovanni and Iaderola, Iacopo and Paoloni, Andrea and Todisco, Massimiliano},
    booktitle={International Conference on Language Resources and Evaluation (LREC 2014)},
    pages={3501--3504},
    year={2014},
    organization={European Language Resources Association (ELRA)}
}
"""

_HOMEPAGE = 'http://voice.fub.it/activities/corpora/emovo/index.html'

LABEL_MAP = {
    'rab': 'anger',
    'pau': 'fear',
    'gio': 'happiness',
    'tri': 'sadness',
    'neu': 'neutral',
}

_SAMPLE_RATE = 48000

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

    print(items_and_groups)

    groups = sorted(set(group_id for item_id, group_id in items_and_groups))
    rng = np.random.RandomState(split_number)
    rng.shuffle(groups)

    print(len(groups))

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

class Emovo(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for emovo dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain the file emovo.zip.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'audio': Audio(file_format='wav', sample_rate=_SAMPLE_RATE),
                'label': tfds.features.ClassLabel(names=LABEL_MAP.values()),
                'speaker_id': tf.string
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('audio', 'label'),    # Set to `None` to disable
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        zip_path = os.path.join(dl_manager.manual_dir, 'emovo.zip')

        if not tf.io.gfile.exists(zip_path):
            raise AssertionError(
                'EMOVO requires manual download of the data. Please download '
                'the audio data and place it into: {}'.format(zip_path))

        extract_path = dl_manager.extract(zip_path)

        items_and_groups = []
        for fname in tf.io.gfile.glob('{}/*/*/*.wav'.format(extract_path)):
            if os.path.basename(fname).split("-")[0] in ['rab', 'anger', 'pau', 'gio', 'tri', 'neu']:
                speaker_id = parse_name(os.path.basename(fname), from_i=4, to_i=6)
                items_and_groups.append((fname, speaker_id))

        # Like SAVEE (https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/savee.py)
        split_probs = [('train', 0.6), ('validation', 0.2), ('test', 0.2)]

        splits = _get_inter_splits_by_group(items_and_groups, split_probs, 0)

        with open("train.lst", 'w') as f:
            for l in splits['train']:
                f.write(l)

        with open("val.lst", 'w') as f:
            for l in splits['validation']:
                f.write(l)

        with open("test.lst", 'w') as f:
            for l in splits['test']:
                f.write(l)


        # Returns the Dict[split names, Iterator[Key, Example]]
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'file_names': splits['train']},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={'file_names': splits['validation']},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={'file_names': splits['test']},
            ),
        ]

    def _generate_examples(self, file_names):
        """Yields examples."""
        # Yields (key, example) tuples from the dataset
        for fname in file_names:
            wavname = os.path.basename(fname)
            speaker_id = parse_name(wavname, from_i=4, to_i=6)
            label = parse_name(wavname, from_i=0, to_i=3, mapping=LABEL_MAP)
            example = {'audio': fname, 'label': label, 'speaker_id': speaker_id}
            yield fname, example
