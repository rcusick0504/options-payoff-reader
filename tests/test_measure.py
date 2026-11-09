"""Numerical recovery must match equations, and missing evidence must abstain."""
import numpy as np
import pytest
from PIL import Image
from src.measure import number, calibrate, fit_curve, measure
from src.payoffs import construct, profit


@pytest.mark.parametrize("label,strikes,roots,maximum,loss", [
    (0,[100],[105],"unbounded",5),
    (1,[100],[95],95,5),
    (2,[90,110],[95],15,5),
    (3,[90,110],[105],15,5),
    (4,[100],[90,110],"unbounded",10),
    (5,[90,110],[80,120],"unbounded",10),
    (6,[80,100,120],[85,115],15,5),
    (7,[100],[90,110],10,"unbounded"),
])
def test_numeric_fit(label,strikes,roots,maximum,loss):
    x=np.linspace(40,160,501)
    result=fit_curve(x,profit(x,construct(label)),label)
    assert result is not None
    np.testing.assert_allclose(result["strikes"],strikes,atol=.1)
    np.testing.assert_allclose(result["breakevens"],roots,atol=.1)
    for key,expected in [("max_profit",maximum),("max_loss",loss)]:
        if expected=="unbounded":
            assert result[key]==expected
        else:
            assert result[key]==pytest.approx(expected,abs=.1)


def test_no_guessed_values():
    result=measure(Image.new("RGB",(400,300),"white"),0,dict(available=True,tokens=[],message=""))
    assert not result["available"] and result["values"] is None
    assert number("−$1,200.5")==-1200.5
    assert number("10%") is None
    assert calibrate([])==(None,None)


def test_non_linear_ticks_are_rejected():
    ticks=[dict(text=str(v),confidence=1,box=[.1,.2+i*.2,.15,.22+i*.2]) for i,v in enumerate([100,10,1])]
    assert calibrate(ticks)[1] is None


def test_wrong_shape_is_rejected():
    x=np.linspace(40,160,501)
    assert fit_curve(x,profit(x,construct(6)),0) is None
