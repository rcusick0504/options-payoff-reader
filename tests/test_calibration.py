import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.calibration import manual_axes
from src.measure import calibrate, measure
from src.payoffs import construct, profit


def points():
    return [dict(axis="x",position=[.1,.85],value=40),dict(axis="x",position=[.9,.85],value=160),
            dict(axis="y",position=[.08,.8],value=-5),dict(axis="y",position=[.08,.2],value=55)]


def test_confirmed_axes_and_values_work_without_ocr():
    x=np.linspace(40,160,481)
    y=profit(x,construct(0))
    im=Image.new("RGB",(640,480),"white")
    ImageDraw.Draw(im).line(list(zip((.1+(x-40)/120*.8)*640,(.2+(55-y)/60*.6)*480)),fill="#245a9a",width=3)
    result=measure(im,0,ocr=dict(available=False,tokens=[],message="OCR unavailable"),vertical_axis="profit_loss",axis_points=points())
    assert result["available"]
    assert result["values"]["strikes"]==pytest.approx([100],abs=.3)
    assert result["values"]["breakevens"]==pytest.approx([105],abs=.3)
    assert result["values"]["max_loss"]==pytest.approx(5,abs=.3)
    assert result["axis_evidence"]["x"]["source"]=="user"
    payoff=measure(im,0,ocr=dict(available=False,tokens=[],message=""),vertical_axis="payoff",axis_points=points())
    assert payoff["values"]["max_loss"] is None
    assert payoff["values"]["breakevens"]==[]


@pytest.mark.parametrize("bad", [None,{},[dict(axis="z",position=[.1,.2],value=5)],
    [dict(axis="x",position=[.1,.2],value=float("inf"))],
    [dict(axis="x",position=[2,.2],value=5)],
    [dict(axis="x",position=[.1,.2],value=True)],
    [dict(axis="x",position=[.1,.2],value=5)]])
def test_invalid_ticks_are_rejected(bad):
    with pytest.raises(ValueError):
        manual_axes(bad)


def test_inconsistent_ticks_are_rejected():
    bad=points()[:2]+[dict(axis="x",position=[.5,.85],value=150)]
    with pytest.raises(ValueError,match="linear scale"):
        manual_axes(bad)


def test_two_staggered_price_labels_calibrate_without_using_vertical_labels():
    tokens=[]
    for value,y in [(4,.1),(2,.3),(0,.5),(-2,.7)]:
        tokens.append(dict(text=str(value),confidence=1,box=[.1,y-.015,.15,y+.015]))
    for value,x,y in [(100,.4,.55),(105,.65,.5)]:
        tokens.append(dict(text=str(value),confidence=1,box=[x-.02,y-.015,x+.02,y+.015]))
    xa,ya=calibrate(tokens)
    assert xa is not None and ya is not None
    assert xa["slope"]*.4+xa["intercept"]==pytest.approx(100)
    assert len(xa["ticks"])==2
