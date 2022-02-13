# ConvNeXt

Implementation of [A ConvNet for the 2020s][convnext] by Liu, et al., in [Haiku].

The authors limited themselves to a choice of architecture whose FLOP counts at various spatial resolutions could be easily compared with their baselines, in the interest of accurately representing the improvements resulting from numerous architectural changes in their experiments. For my use-case I saw fit to replace this default configuration with one given by Brock, et al. in [High-Performance Large-Scale Image Recognition Without Normalization][nfnets], which is instead motivated by a strong tradeoff observed between representational capacity and training latency during empirical architecture search.

## API

The interface uses layer-wise normalization as employed in ConvNeXt by default and includes support for substituting other normalization schemes, including the baseline batch normalization. Specifying `norm=None` in model constructor will result in "norm-free" weight standardization as employed in NFNets.

```python
import haiku as hk
import jax

from convnext import ConvNeXt


def ConvNeXtTiny(inputs, is_training=True):
    cnn = ConvNeXt(depths=[3, 3, 9, 3])
    logits = cnn(inputs, is_training)
    return logits


rng = hk.PRNGSequence(42)
inputs = jax.random.truncated_normal(next(rng), -1, 1, (1, 224, 224, 3))
params, state = hk.transform_with_state(ConvNeXtTiny).init(next(rng), inputs)
logits = hk.transform_with_state(ConvNeXtTiny).apply(params, state, next(rng), inputs)
```

[convnext]: https://arxiv.org/abs/2201.03545
[nfnets]: https://arxiv.org/abs/2102.06171
[haiku]: httpS://github.com/deepmind/dm-haiku
