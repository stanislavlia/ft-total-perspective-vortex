from .run import run_train_mode, run_predict_mode
from .evaluate import run_evaluate, ExperimentResults, EXPERIMENTS
from .visualize import run_visualize

#We can import all our defined utils (run_something) from the module, without specifying submodules
#like this from cli_commands import run_train_mode, run_predict_mode, run_evaluate, run_visualize

__all__ = [
    "run_train_mode",
    "run_predict_mode",
    "run_evaluate",
    "run_visualize",
    "ExperimentResults",
    "EXPERIMENTS",
]
