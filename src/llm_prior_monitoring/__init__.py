"""LLM-prior monitoring: sequential changepoint detection via interpretable concepts."""

from .changepoint import (
    ChangepointLearner,
    KernelCPDLearner,
    MeanShiftChangepointLearner,
    SlicedWassersteinCPDLearner,
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
from data import embed, load_one_liners
from .e_detector import (
    EDetector,
    SharedLEDetector,
)
from .e_detector_sr import (
    SymmetricSREDetector,
)
from .e_process import EProcess, SymmetricEProcess
from .kernels import (
    median_heuristic_bandwidth,
    mmd_squared,
    pairwise_dist,
    rbf_kernel,
)
from .metrics import (
    detection_accounting,
    edd_curve,
    first_crossing,
    matched_at_arl,
    null_run_length_curve,
)
from .mmd_detector import SlidingMMDDetector, calibrate_threshold
from .monitor import AdaptiveMonitoring
from .synthetic import (
    SyntheticConfig,
    sample_p0,
    sample_p1,
    true_shift_features,
)
from .utils import (
    bootstrap_ci,
    format_or_na,
    nanmean_or_nan,
    parse_float_list,
    write_rows,
)

__all__ = [
    # changepoint
    "ChangepointLearner",
    "KernelCPDLearner",
    "MeanShiftChangepointLearner",
    "SlicedWassersteinCPDLearner",
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
    # data
    "build_diluted_pool",
    "build_stream",
    "embed",
    "load_one_liners",
    # e_detector
    "EDetector",
    "SharedLEDetector",
    # e_detector_sr
    "SymmetricSREDetector",
    # e_process
    "EProcess",
    "SymmetricEProcess",
    # kernels
    "median_heuristic_bandwidth",
    "mmd_squared",
    "pairwise_dist",
    "rbf_kernel",
    # metrics (selective re-export)
    "detection_accounting",
    "edd_curve",
    "first_crossing",
    "matched_at_arl",
    "null_run_length_curve",
    # mmd_detector
    "SlidingMMDDetector",
    "calibrate_threshold",
    # monitor
    "AdaptiveMonitoring",
    # synthetic
    "SyntheticConfig",
    "sample_p0",
    "sample_p1",
    "true_shift_features",
    # utils
    "bootstrap_ci",
    "format_or_na",
    "nanmean_or_nan",
    "parse_float_list",
    "write_rows",
]
