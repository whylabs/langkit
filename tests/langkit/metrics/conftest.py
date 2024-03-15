from langkit.core.workflow import Workflow
from langkit.metrics.library import lib

wf = Workflow([lib.presets.all()])
