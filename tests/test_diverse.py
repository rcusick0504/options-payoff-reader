import numpy as np
from src.diverse import sample


def test_families_and_labels_are_crossed_and_separated():
    families={s:set() for s in ["train","unseen_val","unseen_test"]}
    for split,count in [("train",50),("unseen_val",5),("unseen_test",5)]:
        for f in range(count):
            im,label,meta=sample(f*8,split)
            assert label==0 and min(im.size)>=280
            families[split].add(meta["family"])
    assert len(families["train"])==50
    assert not families["train"] & families["unseen_val"]
    assert not families["unseen_val"] & families["unseen_test"]


def test_diverse_reproducible():
    a,label,meta=sample(10)
    b,_,_=sample(10)
    assert np.array_equal(a,b)
