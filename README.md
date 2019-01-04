# NIPS2018_DECAPROP
Implementation of Densely Connected Attention Propagation for Reading Comprehension (NIPS 2018) - Yi Tay, Luu Anh Tuan, Siu Cheung Hui, Jian Su.

This model achieves quite competitive performance on four RC benchmarks (SearchQA, NewsQA, Quasar and NarrativeQA).

https://arxiv.org/abs/1811.04210

### Model Code
The general idea here is that `./model/span_model.py` contains the main span model and `./model/decaprop.py` contains the DecaProp implementation. Bidirectional Attention Connectors (BAC) implementation is found at `./tylib/lib/att_op.py`.

```python
from tylib.lib.att_op import bidirectional_attention_connector
# c and q are sequences of bsz x seq_len x dim.
# seq_len may be different
# the output ff is the propagated feature.
c, q, ff = bidirectional_attention_connector(
                  c, q, c_len, q_len,
                  None, None,
                  mask_a=cmask, mask_b=qmask,
                  initializer=self.init,
                  factor=32, factor2=32,
                  name='bac')
```

### Prep Scripts

You may find them at `./prep/` where datasets such as Squad, NewsQA, SearchQA and Quasar are found. Many of our pre-processing scripts reference https://github.com/nusnlp/amanda. Open domain QA dataset preprocessing were obtained from https://github.com/shuohangwang/mprc (reinforced reader ranker codebase by Wang et al.)

Please make a directory named `./corpus/` (for hosting raw datasets) and `./datasets/` for hosting prep-ed files. The key idea is that we prep the dataset into an `env.gz` file for training/evaluation.

## Notes and Disclaimer

Most of the relevant code have been uploaded to this repository. I currently do not have the GPU resources to re-validate this repository. Assuming I didn't accidentally omit any code (while copying from my main repository and removing irrelevant/WIP code), this repository should run fine (the entry point is `train_span.py`, more running notes will be added when I have time).

The arguments in the argparser **do not** represent the optimal hyperparameters (from the time of NIPS'18 experiments, many other experiments were conducted, which may have changed the default hyperparameters). However, just couple of weeks ago I managed to get similar scores for searchqa/quasar.

Another useful note is that i use a language-based compositional control for model architecture, using `if` statements and keyword to control which the graph construction. This is controlled by `--rnn_type` in argparser. Also note that due to some tensorflow version upgrade issues, the cudnn CoVe LSTM is not working for the time being.

## References

If you find our repository useful, please cite our paper:

```
@article{DBLP:journals/corr/abs-1811-04210,
  author    = {Yi Tay and
               Luu Anh Tuan and
               Siu Cheung Hui and
               Jian Su},
  title     = {Densely Connected Attention Propagation for Reading Comprehension},
  journal   = {CoRR},
  volume    = {abs/1811.04210},
  year      = {2018},
  url       = {http://arxiv.org/abs/1811.04210},
  archivePrefix = {arXiv},
  eprint    = {1811.04210},
  timestamp = {Fri, 23 Nov 2018 12:43:51 +0100},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1811-04210},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgements

Several useful code bases we used in our work:

1. https://github.com/HKUST-KnowComp/R-Net (for cudnn RNNs and base R-NET model)
2. https://github.com/nusnlp/amanda (thanks for the evaluators and preprocessors which were useful!)
3. https://github.com/shuohangwang/mprc (For preprocessing of searchqa and quasar!)
