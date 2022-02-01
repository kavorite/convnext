import inspect
from functools import partial

import haiku as hk
import jax

from . import nf


def _channels(x0, d=8):
    x = max(d, int(x0 + d / 2) // d * d)
    if x < 0.9 * x0:
        x += d
    return x


LayerNorm = partial(hk.LayerNorm, axis=-1, create_scale=True, create_offset=True)


class NormDispatcher(hk.Module):
    def __init__(self, inner=None, name=None):
        super().__init__(name=name)
        if inner is not None:
            self.inner = inner()
        else:
            self.inner = None

    def __call__(self, inputs, is_training):
        if self.inner is not None:
            if "is_training" in inspect.signature(self.inner).parameters:
                logits = self.inner(inputs, is_training=is_training)
            else:
                logits = self.inner(inputs)
        else:
            logits = inputs
        return logits


class Block(hk.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        kernel_shape=7,
        norm=LayerNorm,
        act=jax.nn.gelu,
        sdrate=0.2,
        name=None,
    ):
        super().__init__(name=name)
        self.kernel_shape = kernel_shape
        self.which_conv = nf.Conv2D if norm is None else hk.Conv2D
        self.sdrate = sdrate
        self.ch_oup = out_channels
        self.ch_hdn = hidden_channels
        self.norm = NormDispatcher(norm)
        self.act = act

    def __call__(self, inputs, is_training):
        drop = is_training and jax.random.uniform(hk.next_rng_key()) >= self.sdrate
        b_l = jax.lax.cond(drop, lambda: 1.0, lambda: 0.0)
        logits = self.which_conv(
            self.ch_oup,
            kernel_shape=self.kernel_shape,
            feature_group_count=self.ch_oup,
        )(inputs)
        logits = self.norm(logits, is_training)
        logits = self.which_conv(_channels(self.ch_hdn), kernel_shape=1)(logits)
        logits = self.act(logits)
        logits = self.which_conv(_channels(self.ch_oup), kernel_shape=1)(logits)
        return inputs + b_l * logits


class Downsampling(hk.Module):
    def __init__(self, width, norm=LayerNorm, name=None):
        super().__init__(name=name)
        self.norm = NormDispatcher(norm)
        self.width = width

    def __call__(self, inputs, is_training):
        logits = hk.Conv2D(self.width, kernel_shape=2, stride=2)(inputs)
        logits = self.norm(logits, is_training)
        return logits


class ConvNeXt(hk.Module):
    def __init__(
        self,
        pool=partial(hk.AvgPool, window_shape=7, strides=7, padding="SAME"),
        head=partial(hk.Linear, 1000),
        norm=LayerNorm,
        act=jax.nn.gelu,
        widths=[96, 192, 384, 384],
        expand_factors=[4.0] * 4,
        depths=[1, 2, 6, 3],
        sdrate=0.2,
        name=None,
    ):
        super().__init__(name=name)
        assert (
            len(expand_factors) == len(widths) == len(depths)
        ), "widths, depths, and expansions must all describe the same number of stages"
        self.stem = hk.Conv2D(widths[0], kernel_shape=4, stride=4, name="stem_conv")
        self.norm = NormDispatcher(norm, name="stem_norm")
        self.pool = pool
        self.head = head
        self.layers = []
        block_count = sum(depths)
        block_index = 0

        for i, (width, depth, expansion) in enumerate(
            zip(widths, depths, expand_factors)
        ):
            if i != 0:
                downsampling = Downsampling(
                    width, norm=norm, name=f"stage_{i}_downsampling"
                )
                self.layers.append(downsampling)
            for j in range(depth):
                block = Block(
                    width,
                    width * expansion,
                    norm=norm,
                    act=act,
                    sdrate=sdrate * block_index / block_count,
                    name=f"stage_{i}_block_{j}",
                )
                self.layers.append(block)
                block_index += 1

    def __call__(self, inputs, is_training):
        batchc = inputs.shape[0]
        logits = self.stem(inputs)
        logits = self.norm(logits, is_training)
        for layer in self.layers:
            logits = layer(logits, is_training)
        if self.pool is not None:
            logits = self.pool()(logits)
        if self.head is not None:
            logits = logits.reshape(batchc, -1)
            logits = self.head()(logits)
        return logits
