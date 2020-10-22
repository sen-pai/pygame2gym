from setuptools import setup

setup(
    name="simple_discrete_game",
    version="0.0.1",
    description="Simple Pygame to OpenAI Gym environment",
    author="Sharan Pai",
    url="https://github.com/sen-pai/pygame2gym",
    install_requires=["gym", "numpy", "pygame"],
    python_requires=">=3.6",
)
