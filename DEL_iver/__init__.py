# DEL_iver/__init__.py

#Any function listed here will be callable with no imports needed other then 
# Import everything you want available at the top level when you run import DEL_iver
from .data_loader.data_reader import DataReader
from .data_loader.Split_TestTrainVal import split_data
from .molecules.Make_BBdictionaries import generate_bb_dictionaries
from .molecules.ECFP4calculator import gen_fingerprints 
from .analysis.compute import compute_pbind_and_enrichment
from .analysis.compute import find_best_bb
from .analysis.compute import find_best_disynthon
from .analysis.compute import data_set_statistics
from .analysis.plotting import draw_disynthons
from .analysis.plotting import draw_bb
from .analysis.plotting import plot_disynthons
from .analysis.compute import compute_chemical_descriptors
from .analysis.plotting import plot_bb
from .models.trainmodels import train_default
from .models.trainmodels import train_invariant
from .models.inference_test import inference
from .analogs.embed_analogs import analog_embed
from .analogs.inference_analogs import inference_analog_moles



import warnings

def _custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"\n[DEL-iver] {category.__name__}: {message}\n"

warnings.formatwarning = _custom_warning_format