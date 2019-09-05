from ..staph.analysis.compute_chisq import sse_chisq


def test_sse_chisq():
    sse2par = 5
    sse1par = 5.01
    x = sse_chisq(sse_2par=sse2par, sse_1par=sse1par, sd_estimate=1)
    assert x[1] == "Unable to reject simpler model"
    sse1par = 1e5
    x = sse_chisq(sse_2par=sse2par, sse_1par=sse1par, sd_estimate=1)
    assert x[1] == "Reject simpler model"
