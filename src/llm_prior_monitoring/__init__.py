"""LLM-prior monitoring: sequential changepoint detection via interpretable concepts."""

from .changepoint import (
    ChangepointLearner,
    KernelCPDLearner,
    MeanShiftChangepointLearner,
    SubsetScanChangepointLearner,
)
from .concept_selectors import (
    MockConceptSelector,
    MockConceptState,
    RankedConceptSelector,
)
from .conformal_martingale import (
    SCORER_REGISTRY,
    CentroidScore,
    ConformalPValue,
    ConformalSRDetector,
    ConformalTestMartingaleMonitor,
    ConformityScorer,
    JumperMartingale,
    NearestNeighborScore,
)
from .ctm_config import CTMConfig
from .feature_extractors import (
    MockSparseFeatureExtractor,
    SingleConceptSparseFeatureExtractor,
    SparseExtractionMatrix,
    SparseFeatureExtractor,
)
from .mock_HACHI import MockConceptGenerator, MockKeyphrase
from .e_detector import (
    EDetector,
    SharedLEDetector,
)
from .e_detector_sr import (
    SymmetricSREDetector,
)
from .e_process import EProcess, SymmetricEProcess
from .metrics import (
    detection_accounting,
    first_crossing,
)
from .monitor import AdaptiveMonitoring
from .synthetic import (
    SyntheticConfig,
    sample_p0,
    sample_p1,
    true_shift_features,
)
from .utils import (
    bootstrap_ci,
    parse_float_list,
    write_rows,
)

__all__ = [
    # changepoint
    "ChangepointLearner",
    "KernelCPDLearner",
    "MeanShiftChangepointLearner",
    "SubsetScanChangepointLearner",
    # concept_selectors
    "MockConceptSelector",
    "MockConceptState",
    "RankedConceptSelector",
    # conformal_martingale
    "SCORER_REGISTRY",
    "CentroidScore",
    "ConformalPValue",
    "ConformalSRDetector",
    "ConformalTestMartingaleMonitor",
    "ConformityScorer",
    "JumperMartingale",
    "NearestNeighborScore",
    # ctm_config
    "CTMConfig",
    # mock HACHI subcomponents
    "MockConceptGenerator",
    "MockKeyphrase",
    # mock feature extractors
    "MockSparseFeatureExtractor",
    "SingleConceptSparseFeatureExtractor",
    "SparseExtractionMatrix",
    "SparseFeatureExtractor",
    # e_detector
    "EDetector",
    "SharedLEDetector",
    # e_detector_sr
    "SymmetricSREDetector",
    # e_process
    "EProcess",
    "SymmetricEProcess",
    # metrics (selective re-export)
    "detection_accounting",
    "first_crossing",
    # monitor
    "AdaptiveMonitoring",
    # synthetic
    "SyntheticConfig",
    "sample_p0",
    "sample_p1",
    "true_shift_features",
    # utils
    "bootstrap_ci",
    "parse_float_list",
    "write_rows",
]
