import sys

from langkit.core.workflow import EvaluationWorkflow
from langkit.metrics.library import lib

if __name__ == "__main__":
    wf = EvaluationWorkflow(metrics=[lib.all_metrics()], cache_assets="--skip-downloads" not in sys.argv)
    # Run it to ensure nothing else ends up getting lazily cached
    wf.run({"prompt": "How are you today?", "response": "I'm doing great!"})
