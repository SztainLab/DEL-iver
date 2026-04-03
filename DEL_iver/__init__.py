# DEL_iver/__init__.py

#Any function listed here will be callable with no imports needed other then 
# Import everything you want available at the top level when you run import DEL_iver
from .data_loader.data_reader import DataReader
from .data_loader.Split_TestTrainVal import split_data
#from .molecules.Make_BBdictionaries import generate_BB_dictionaries



import warnings

def _custom_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"\n[DEL-iver] {category.__name__}: {message}\n"

warnings.formatwarning = _custom_warning_format