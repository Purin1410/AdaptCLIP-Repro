from .logger import get_logger
from .utils import get_transform, normalize, setup_seed

# Modified 2026-10-19: keep optional evaluation dependencies out of training.
def __getattr__(name):
    if name == 'Evaluator':
        from .effecient_metric import Evaluator
        return Evaluator
    if name == 'visualizer':
        from .visualization import visualizer
        return visualizer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
