from setuptools import setup

setup(
    name="convnext",
    version="1.0",
    description="ConvNeXt for the 2020s in JAX",
    url="https://github.com/kavorite/convnext",
    author="kavorite",
    license="MIT",
    package_dir={"": "src"},
    packages=["convnext"],
    install_requires=[
        "chex>=0.1.0",
        "dm_haiku>=0.0.5",
        "jax>=0.2.27",
        "optax>=0.0.9",
        "tqdm>=4.62.3",
    ],
)
