# Imports for registering policies
from policies.nf_exp import NearestFrontierExp
from policies.pf_exp import PFExp
from policies.policy_registry import policy_registry

# SemExp is only used by the Gibson evaluation path and requires detectron2.
# Check before importing it so a failed import cannot leave semexp partially
# initialized and interfere with MP3D's PFExp + RedNet path.
from importlib.util import find_spec

if find_spec("detectron2") is not None:
    from policies.sem_exp import SemExp
