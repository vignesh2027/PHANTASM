"""
PHANTASM: Probabilistic Hallucination-Aware Neural Transformation
           with Adaptive Synthesis Method

The first ML framework to mathematically invert LLM "failures" into features.

Author: Vignesh S <applemacbook6sep2004@gmail.com>
License: Apache 2.0
"""

__version__ = "1.1.0"
__author__ = "Vignesh S"
__email__ = "applemacbook6sep2004@gmail.com"
__license__ = "Apache-2.0"

from phantasm.core.hgt import HallucinationGradientTracer
from phantasm.core.cmn import ConfabulationMiningNetwork
from phantasm.core.uc import UncertaintyCrystallizer
from phantasm.core.pipeline import PHANTASMPipeline

__all__ = [
    "HallucinationGradientTracer",
    "ConfabulationMiningNetwork",
    "UncertaintyCrystallizer",
    "PHANTASMPipeline",
]
