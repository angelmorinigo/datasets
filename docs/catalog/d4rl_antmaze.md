<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>
  <meta itemprop="name" content="d4rl_antmaze" />
  <meta itemprop="description" content="D4RL is an open-source benchmark for offline reinforcement learning. It provides&#10;standardized environments and datasets for training and benchmarking algorithms.&#10;&#10;To use this dataset:&#10;&#10;```python&#10;import tensorflow_datasets as tfds&#10;&#10;ds = tfds.load(&#x27;d4rl_antmaze&#x27;, split=&#x27;train&#x27;)&#10;for ex in ds.take(4):&#10;  print(ex)&#10;```&#10;&#10;See [the guide](https://www.tensorflow.org/datasets/overview) for more&#10;informations on [tensorflow_datasets](https://www.tensorflow.org/datasets).&#10;&#10;" />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/d4rl_antmaze" />
  <meta itemprop="sameAs" content="https://sites.google.com/view/d4rl/home" />
  <meta itemprop="citation" content="@misc{fu2020d4rl,&#10;    title={D4RL: Datasets for Deep Data-Driven Reinforcement Learning},&#10;    author={Justin Fu and Aviral Kumar and Ofir Nachum and George Tucker and Sergey Levine},&#10;    year={2020},&#10;    eprint={2004.07219},&#10;    archivePrefix={arXiv},&#10;    primaryClass={cs.LG}&#10;}" />
</div>

# `d4rl_antmaze`


Note: This dataset was added recently and is only available in our
`tfds-nightly` package
<span class="material-icons" title="Available only in the tfds-nightly package">nights_stay</span>.

*   **Description**:

D4RL is an open-source benchmark for offline reinforcement learning. It provides
standardized environments and datasets for training and benchmarking algorithms.

*   **Config description**: See more details about the task and its versions in
    https://github.com/rail-berkeley/d4rl/wiki/Tasks#antmaze

*   **Homepage**:
    [https://sites.google.com/view/d4rl/home](https://sites.google.com/view/d4rl/home)

*   **Source code**:
    [`tfds.d4rl.d4rl_antmaze.D4rlAntmaze`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/d4rl/d4rl_antmaze/d4rl_antmaze.py)

*   **Versions**:

    *   `1.0.0`: Initial release.
    *   **`1.1.0`** (default): No release notes.

*   **Download size**: `Unknown size`

*   **Dataset size**: `Unknown size`

*   **Auto-cached**
    ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)):
    Unknown

*   **Splits**:

Split | Examples
:---- | -------:

*   **Features**:

```python
FeaturesDict({
    'steps': Dataset({
        'action': Tensor(shape=(8,), dtype=tf.float32),
        'discount': tf.float32,
        'infos': FeaturesDict({
            'goal': Tensor(shape=(2,), dtype=tf.float32),
            'qpos': Tensor(shape=(15,), dtype=tf.float32),
            'qvel': Tensor(shape=(14,), dtype=tf.float32),
        }),
        'is_first': tf.bool,
        'is_last': tf.bool,
        'is_terminal': tf.bool,
        'observation': Tensor(shape=(29,), dtype=tf.float32),
        'reward': tf.float32,
    }),
})
```

*   **Supervised keys** (See
    [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)):
    `None`

*   **Figure**
    ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)):
    Not supported.

*   **Examples**
    ([tfds.as_dataframe](https://www.tensorflow.org/datasets/api_docs/python/tfds/as_dataframe)):
    Missing.

*   **Citation**:

```
@misc{fu2020d4rl,
    title={D4RL: Datasets for Deep Data-Driven Reinforcement Learning},
    author={Justin Fu and Aviral Kumar and Ofir Nachum and George Tucker and Sergey Levine},
    year={2020},
    eprint={2004.07219},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


## d4rl_antmaze/umaze-v0 (default config)

## d4rl_antmaze/umaze-diverse-v0

## d4rl_antmaze/medium-play-v0

## d4rl_antmaze/medium-diverse-v0

## d4rl_antmaze/large-diverse-v0

## d4rl_antmaze/large-play-v0