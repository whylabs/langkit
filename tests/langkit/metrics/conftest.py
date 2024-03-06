from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.library import lib

wf = EvaluationWorkflow([lib.presets.all()])
