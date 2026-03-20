# gen_joint_mtide/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("gen-joint-mtide")
except PackageNotFoundError:
    # package is not installed (e.g., running from source)
    __version__ = "0.0.0"


# core module
from .core.emg_arbitrary_variance import *
from .core.evaluate_gen_models import *
from .core.gen_model_strategies import *
from .core.simulate import *

# If you want to include R scripts in Python (via rpy2)
# from .core.fit_tcopula import *   # only if you wrap functions with rpy2
# from .core.gen_t_copula import *  # same

# optionally, expose data files
import os

data_dir = os.path.join(os.path.dirname(__file__), "data")
example_data = os.path.join(data_dir, "example_data.csv")

__all__ = [
    # Python functions
    *[name for name in dir() if not name.startswith("_")],
    "example_data",
]
