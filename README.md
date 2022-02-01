# ConvNeXt

Implementation of [A ConvNet for the 2020s][convnext] in [haiku]. Bottleneck stages are substituted for those found in [High-Performance Large-Scale Image Recognition Without Normalization][nfnets], the arrangements of which are better empirically motivated than those of the ConvNeXt authors, who favored architectures with computational load roughly comparable to their baselines in order to generate experimentally valid results. 

Interface supports ConvNeXt LayerNorm in addition to baseline BatchNorm. Specifying `norm=None` in model constructor will result in "norm-free" weight standardization as employed in NFNets. 

[convnext]: https://arxiv.org/abs/2201.03545
[nfnets]: https://arxiv.org/abs/2102.06171
[haiku]: httpS://github.com/deepmind/dm-haiku
