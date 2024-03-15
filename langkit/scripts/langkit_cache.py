import sys

from langkit.core.workflow import Workflow
from langkit.metrics.library import lib

if __name__ == "__main__":
    wf = Workflow(metrics=[lib.presets.all()], cache_assets="--skip-downloads" not in sys.argv)
    # Run it to ensure nothing else ends up getting lazily cached
    wf.run({"prompt": "How are you today?", "response": "I'm doing great!"})
