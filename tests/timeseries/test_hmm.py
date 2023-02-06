from stochproc.timeseries import HiddenMarkovModel
import pytest as pt
import torch


from .test_affine import SAMPLES
from .constants import BATCH_SHAPES


class TestHMM(object):
    @pt.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_hmm(self, batch_shape):
        probs = torch.tensor([[0.35, 0.65], [0.1, 0.9]])

        hmm = HiddenMarkovModel((probs,))
        samples = hmm.expand(batch_shape).sample_states(SAMPLES).get_path()

        assert samples.shape == torch.Size([SAMPLES]) + batch_shape + hmm.event_shape
