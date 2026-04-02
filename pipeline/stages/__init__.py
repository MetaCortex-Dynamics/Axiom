"""Pipeline stages S1-S5 per SPEC-PIPELINE-001 Part B."""

from .s1_segment import segment
from .s2_classify import classify
from .s3_decompose import decompose
from .s4_validate import validate_and_score
from .s5_emit import emit
