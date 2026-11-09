from PIL import Image
from src import ocr


def test_portable_ocr_boxes_map_back_to_original_coordinates(monkeypatch):
    def engine(pixels):
        assert pixels.shape==(150,300,3)
        assert tuple(pixels[0,0])==(30,20,10)
        return ([[[[30,45],[60,45],[60,75],[30,75]],"100",.95]],None)
    monkeypatch.setattr(ocr,"rapid_engine",lambda:engine)
    result=ocr.rapid_ocr(Image.new("RGB",(100,50),(10,20,30)))
    assert result["available"]
    assert result["tokens"][0]["box"]==[.1,.3,.2,.5]


def test_optional_ocr_dependency_can_be_missing(monkeypatch):
    def absent():
        raise ImportError("Not installed")
    monkeypatch.setattr(ocr,"rapid_engine",absent)
    monkeypatch.setattr(ocr,"apple_ocr",lambda *args,**kwargs:dict(available=False,tokens=[],message="Unavailable"))
    result=ocr.recognize(Image.new("RGB",(100,50)))
    assert not result["available"] and result["tokens"]==[]
