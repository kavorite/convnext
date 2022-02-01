# ConvNeXt

Implementation of [A ConvNet for the 2020s][convnext] in [haiku]. Default stage depths are substituted for those found in [High-Performance Large-Scale Image Recognition Without Normalization][nfnets], the arrangements of which are somewhat better motivated than those of the ConvNeXt authors, who favored architectures with computational load roughly comparable to their baselines in order to generate experimentally valid results. 

Interface supports ConvNeXt LayerNorm in addition to baseline BatchNorm. Specifying `norm=None` in model constructor will result in "norm-free" weight standardization as employed in NFNets. 


```python
import haiku as hk
import jax

from convnext import ConvNeXt


def ConvNextT(inputs, is_training=True):
    cnn = ConvNeXt(depths=[3, 3, 9, 3])
    logits = cnn(inputs, is_training)
    return logits


rng = hk.PRNGSequence(42)
inputs = jax.random.truncated_normal(next(rng), -1, 1, (1, 224, 224, 3))
params, state = hk.transform_with_state(ConvNextT).init(next(rng), inputs)
logits = hk.transform_with_state(ConvNextT).apply(params, state, next(rng), inputs)
```

[convnext]: https://arxiv.org/abs/2201.03545
[nfnets]: https://arxiv.org/abs/2102.06171
[haiku]: httpS://github.com/deepmind/dm-haiku

