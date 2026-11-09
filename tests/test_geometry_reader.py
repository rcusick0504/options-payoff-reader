import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.geometry_reader import read_geometry
from src.payoffs import construct, profit


@pytest.mark.parametrize("label",range(8))
def test_piecewise_reader_without_text_or_fixed_plot_box(label):
    x=np.linspace(40,160,481)
    y=profit(x,construct(label))
    im=Image.new("RGB",(640,480),"#f3f1e9")
    points=list(zip(65+(x-40)*4,370-(y-y.min())/np.ptp(y)*250))
    ImageDraw.Draw(im).line(points,fill="#245a9a",width=4)
    result=read_geometry(im)
    assert result is not None and result["label"]==label


def test_blank_and_curved_inputs_do_not_match():
    blank=Image.new("RGB",(640,480),"white")
    assert read_geometry(blank) is None
    d=ImageDraw.Draw(blank)
    x=np.linspace(60,580,500)
    d.line(list(zip(x,240+120*np.sin((x-60)/520*4*np.pi))),fill="blue",width=3)
    assert read_geometry(blank) is None


def test_missing_plateau_is_not_invented_as_a_v():
    im=Image.new("RGB",(640,480),"white")
    draw=ImageDraw.Draw(im)
    draw.line([(65,100),(265,350)],fill="#245a9a",width=4)
    draw.line([(345,350),(545,100)],fill="#245a9a",width=4)
    assert read_geometry(im) is None
