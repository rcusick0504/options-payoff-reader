"""Read an isolated curve's piecewise geometry independently of chart styling."""
import numpy as np
from scipy.ndimage import label as components, find_objects, median_filter
from .measure import fit_curve, shape


def candidates(image):
    im=image.convert("RGB")
    im.thumbnail((500,500))
    a=np.asarray(im).astype(float)
    h,w=a.shape[:2]
    bg=np.median(a.reshape(-1,3),axis=0)
    # Frequent foreground colors are candidate curve inks, not class features.
    quant=(a.astype(int)//24)*24+12
    colors,counts=np.unique(quant.reshape(-1,3),axis=0,return_counts=True)
    seen=[]
    for color in colors[np.argsort(counts)[::-1]][:24]:
        if np.linalg.norm(color-bg)<65 or any(np.linalg.norm(color-c)<40 for c in seen):
            continue
        seen.append(color)
        mask=np.linalg.norm(a-color,axis=2)<42
        mask[np.mean(mask,axis=1)>.65,:]=False
        mask[:,np.mean(mask,axis=0)>.65]=False
        labels,_=components(mask, structure=np.ones((3,3)))
        kept=np.zeros_like(mask)
        for index,region in enumerate(find_objects(labels),1):
            if region is None:
                continue
            yy,xx=region
            width,height=xx.stop-xx.start,yy.stop-yy.start
            density = np.mean(labels[region]==index)
            if width>w*.16 or (width>w*.04 and height>h*.07 and density<.22):
                kept[region] |= labels[region]==index
        yy,xx=np.where(kept)
        if not len(xx) or np.ptp(xx)<w*.4 or np.ptp(yy)<h*.12:
            continue
        xs,ys=[],[]
        for col in range(xx.min(),xx.max()+1):
            hits=np.flatnonzero(kept[:,col])
            if len(hits) and np.ptp(hits)<h*.07:
                xs.append(col)
                ys.append(-float(np.median(hits)))
        if len(xs)<.75*np.ptp(xx):
            continue
        # Every eligible column must describe one curve; multiple branches reject.
        if np.ptp(xs)<.9*np.ptp(xx):
            continue
        if np.max(np.diff(xs)) > .06*np.ptp(xs):
            # Never bridge an unseen central plateau into an invented V vertex.
            continue
        yield np.array(xs,dtype=float),median_filter(np.array(ys),size=3)


def read_geometry(image):
    matches=[]
    for x,y in candidates(image):
        fits=[]
        for label in range(8):
            fitted=fit_curve(x,y,label)
            if fitted:
                complexity=1 if label in (0,1,4,7) else 2
                fits.append((fitted["fit_error"]+.0003*complexity,label,fitted))
        if not fits:
            continue
        fits.sort(key=lambda f:f[0])
        best=fits[0]
        if best[2]["fit_error"]>.0008:
            continue
        # Require meaningful separation from competing piecewise shapes.
        if len(fits)>1 and fits[1][0]-best[0]<.00015:
            continue
        matches.append(dict(label=best[1],fit_error=best[2]["fit_error"]))
    if not matches or len({m["label"] for m in matches})!=1:
        return None
    return min(matches,key=lambda m:m["fit_error"])
