"""CHIMERA: Calibrated Hierarchical Introspection and Meta-cognitive Error Recognition Assessment.

A comprehensive benchmark for evaluating the meta-cognitive calibration of Large Language Models.

CHIMERA measures:
- Confidence calibration (does stated confidence predict accuracy?)
- Self-error detection (can models identify their own mistakes?)
- Knowledge boundary recognition (do models know what they don't know?)
- Self-correction ability (can models fix errors through introspection?)

Example:
    >>> from chimera import Benchmark, GeminiModel
    >>> model = GeminiModel("gemini-2.0-flash")
    >>> benchmark = Benchmark(model)
    >>> results = benchmark.run()
    >>> print(results.summary())

For more information, see the documentation at:
https://github.com/Rahul-Lashkari/chimera
"""

from chimera.version import __version__

__all__ = [
    "__version__",
]
