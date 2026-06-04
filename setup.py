"""Installation script for the 'agibot_rl_mjlab' python package."""

from setuptools import setup

INSTALL_REQUIRES = [
    "mjlab==1.2.0",
]

setup(
    name="agibot_rl_mjlab",
    packages=["agibot_rl"],
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
