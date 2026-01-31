from sklearn.pipeline import Pipeline
from tqdm import tqdm
from dataclasses import dataclass
from csp import CommonSpatialPattern
from wavelet import HaarWaveletTransform
from constants import Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

@dataclass
class BCIPipelineConfig:
    cv_folds : int = 5
    test_size: float = 0.2
    n_csp_components: int = 6
    use_wavelet: bool = False #BONUS
    wavelet_level: int = 8 #BONUS
    classifier_algorithm: str = "lda" #other options: logreg, svc


def construct_pipeline_from_config(config: BCIPipelineConfig) -> Pipeline:

    #validate
    assert config.cv_folds >= 2
    assert config.wavelet_level > 0
    assert config.n_csp_components > 0
    assert config.test_size < 1 and config.test_size > 0
    assert config.classifier_algorithm in [c.value for c in Classifiers]


    pipeline_components = []

    if config.use_wavelet:
        pipeline_components.append(
            ("haar_wavelet_transform", HaarWaveletTransform(n_levels=config.wavelet_level))
        )
    
    pipeline_components.append(
        ("csp", CommonSpatialPattern(n_components=config.n_csp_components))
    )

    #FOR NOW USE ONLY LDA
    pipeline_components.append(
        ("lda",  LinearDiscriminantAnalysis())
    )

    return Pipeline(steps=pipeline_components)


    


