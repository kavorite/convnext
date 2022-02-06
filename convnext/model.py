import inspect
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp

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


class StochDepth(hk.Module):
    """Batchwise Dropout used in EfficientNet, optionally sans rescaling."""

    def __init__(self, drop_rate, scale_by_keep=False, name=None):
        super().__init__(name=name)
        self.drop_rate = drop_rate
        self.scale_by_keep = scale_by_keep

    def __call__(self, x, is_training) -> jnp.ndarray:
        if not is_training:
            return x
        batch_size = x.shape[0]
        r = jax.random.uniform(hk.next_rng_key(), [batch_size, 1, 1, 1], dtype=x.dtype)
        keep_prob = 1.0 - self.drop_rate
        binary_tensor = jnp.floor(keep_prob + r)
        if self.scale_by_keep:
            x = x / keep_prob
        return x * binary_tensor


class Block(hk.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        kernel_shape=7,
        norm=LayerNorm,
        act=jax.nn.relu,
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
        logits = self.which_conv(
            self.ch_oup,
            kernel_shape=self.kernel_shape,
            feature_group_count=self.ch_oup,
        )(inputs)
        logits = self.norm(logits, is_training)
        logits = self.which_conv(_channels(self.ch_hdn), kernel_shape=1)(logits)
        logits = self.act(logits)
        logits = self.which_conv(_channels(self.ch_oup), kernel_shape=1)(logits)
        logits = StochDepth(self.sdrate)(logits, is_training)
        return self.act(inputs + logits)


class Downsampling(hk.Module):
    def __init__(self, width, norm=LayerNorm, name=None):
        super().__init__(name=name)
        which_conv = nf.Conv2D if norm is None else hk.Conv2D
        self.conv = which_conv(width, kernel_shape=2, stride=2)
        self.norm = NormDispatcher(norm)

    def __call__(self, inputs, is_training):
        logits = self.conv(inputs)
        logits = self.norm(logits, is_training)
        return logits


class ConvNeXt(hk.Module):
    def __init__(
        self,
        head=partial(hk.Linear, 1000),
        pool=hk.avg_pool,
        norm=LayerNorm,
        act=jax.nn.relu,
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
        which_conv = nf.Conv2D if norm is None else hk.Conv2D
        self.stem = which_conv(widths[0], kernel_shape=4, stride=4, name="stem_conv")
        self.norm = NormDispatcher(norm, name="stem_norm")
        self.pool = pool
        self.head = head
        self.blocks = []
        block_count = sum(depths)

        for i, (width, depth, expansion) in enumerate(
            zip(widths, depths, expand_factors)
        ):
            if i != 0:
                downsampling = Downsampling(
                    width, norm=norm, name=f"stage_{i}_downsampling"
                )
                self.blocks.append(downsampling)
            for j in range(depth):
                block = Block(
                    width,
                    width * expansion,
                    norm=norm,
                    act=act,
                    sdrate=sdrate * len(self.blocks) / block_count,
                    name=f"stage_{i}_block_{j}",
                )
                self.blocks.append(block)

    def __call__(self, inputs, is_training):
        batchc = inputs.shape[0]
        logits = self.stem(inputs)
        logits = self.norm(logits, is_training)
        for layer in self.blocks:
            logits = layer(logits, is_training)
        window = strides = (1, *logits.shape[-3:-1], 1)
        if self.pool is not None:
            logits = self.pool(logits, window, strides, padding="SAME")
            logits = logits.reshape(batchc, -1)
        if self.head is not None:
            logits = self.head()(logits)
        return logits
