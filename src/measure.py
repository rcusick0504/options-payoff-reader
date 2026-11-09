"""Conservative local chart digitization from OCR ticks and a single visible curve.

All values are estimates in the chart's axis units. No premiums or contract
multipliers are assumed. Unsupported axes and ambiguous curves return no values.
"""
import re
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import differential_evolution
from .ocr import recognize
from .calibration import manual_axes


def number(text):
    s = text.strip().replace("−", "-").replace("–", "-").replace(",", "").replace("$", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", s):
        return None
    return float(s)


def calibrate(tokens):
    rows = []
    for token in tokens:
        value = number(token["text"])
        if value is not None and token.get("confidence", 0) >= .4:
            l, t, r, b = token["box"]
            rows.append(dict(value=value, x=(l+r)/2, y=(t+b)/2, right=r,
                             text=token["text"], corrected_from=token.get("corrected_from")))
    vertical = None
    def fit(axis):
        candidates = []
        minimum_ticks = 2 if axis == "x" else 3
        minimum_span = .16 if axis == "x" else .3
        for anchor in rows:
            if axis == "x":
                left_edge = max(t["right"] for t in vertical["ticks"]) + .004 if vertical else 0
                group = [p for p in rows if abs(p["y"]-anchor["y"]) < .065 and p["y"] > .35 and p["x"] > left_edge]
            else:
                group = [p for p in rows if abs(p["right"]-anchor["right"]) < .025 and p["right"] < .35]
            if len(group) < minimum_ticks:
                continue
            coords = np.array([p[axis] for p in group])
            values = np.array([p["value"] for p in group])
            if np.ptp(coords) < minimum_span or np.ptp(values) <= 0 or len(np.unique(coords.round(3))) != len(coords):
                continue
            # A merged or misread tick must not contaminate the whole scale.
            consensus = []
            for i in range(len(coords)):
                for j in range(i):
                    if abs(coords[i]-coords[j]) < .08:
                        continue
                    slope = (values[i]-values[j])/(coords[i]-coords[j])
                    intercept = values[i]-slope*coords[i]
                    tolerance = max(.6, abs(slope)*np.ptp(coords)*.025)
                    good = np.abs(slope*coords+intercept-values) < tolerance
                    if good.sum() >= max(minimum_ticks, int(np.ceil(.7*len(coords)))):
                        consensus.append(good)
            if not consensus:
                continue
            good = max(consensus, key=lambda g: int(g.sum()))
            coords, values = coords[good], values[good]
            group = [p for p, keep in zip(group, good) if keep]
            if np.ptp(coords) < minimum_span:
                continue
            a, b = np.polyfit(coords, values, 1)
            error = np.max(np.abs(a*coords+b-values)) / np.ptp(values)
            if error > .025 or (axis == "x" and a <= 0) or (axis == "y" and a >= 0):
                continue
            candidates.append((len(group), -error, dict(slope=float(a), intercept=float(b), ticks=group, error=float(error))))
        return max(candidates, key=lambda c: c[:2])[2] if candidates else None
    vertical = fit("y")
    return fit("x"), vertical


def trace(image, xaxis, yaxis, tokens):
    im = image.convert("RGB")
    im.thumbnail((1000, 1000))
    a = np.asarray(im).astype(float)
    h, w = a.shape[:2]
    left, right = [int(v*w) for v in (min(t["x"] for t in xaxis["ticks"]), max(t["x"] for t in xaxis["ticks"]))]
    # A missed first price label should not crop away a visible left-hand kink.
    ylabel_edge = max(t["right"] for t in yaxis["ticks"])
    left = min(left, int((ylabel_edge + .008)*w))
    ycoords = sorted(t["y"] for t in yaxis["ticks"])
    spacing = float(np.median(np.diff(ycoords)))
    top = int(max(.02, ycoords[0] - spacing*.4)*h)
    # Missing negative tick labels must not clip the loss region of the curve.
    bottom = int((min(t["y"] for t in xaxis["ticks"])-.035)*h)
    if right-left < 30 or bottom-top < 30:
        return None
    crop = a[top:bottom+1, left:right+1]
    bg = np.median(crop.reshape(-1, 3), axis=0)
    delta = np.linalg.norm(crop-bg, axis=2)
    # Color deviation from the background separates colored curves from neutral axes.
    residual = crop - bg
    chroma = np.ptp(residual, axis=2)
    colored = (chroma > 55) & (delta > 65)
    mask = colored if np.mean(colored.any(axis=0)) > .55 else delta > 100
    for token in tokens:
        l,t,r,b = token["box"]
        xl,xr = max(0,int(l*w)-left-2), min(mask.shape[1],int(r*w)-left+3)
        yt,yb = max(0,int(t*h)-top-2), min(mask.shape[0],int(b*h)-top+3)
        if xl < xr and yt < yb:
            mask[yt:yb, xl:xr] = False
    # Remove full-width axes and full-height borders, preserving shorter flat tails.
    mask[np.mean(mask,axis=1) > .85, :] = False
    mask[:, np.mean(mask,axis=0) > .85] = False
    xs, ys = [], []
    for col in range(mask.shape[1]):
        hits = np.flatnonzero(mask[:,col])
        if len(hits) and np.ptp(hits) < mask.shape[0]*.08:
            xs.append((col+left)/w)
            ys.append((np.median(hits)+top)/h)
    if len(xs) < .65*(right-left) or np.ptp(xs) < .85*(right-left)/w:
        return None
    if np.max(np.diff(xs)) > .06*np.ptp(xs):
        return None
    return np.array(xs), median_filter(np.array(ys), size=5)


def shape(x, label, params):
    if label in (0,1,4,7):
        k=params[0]
        return [lambda: np.maximum(x-k,0),lambda: np.maximum(k-x,0),None,None,lambda: np.abs(x-k),None,None,lambda: -np.abs(x-k)][label]()
    lo,hi=sorted(params)
    if label==2:
        return np.clip(x-lo,0,hi-lo)
    if label==3:
        return -np.clip(x-lo,0,hi-lo)
    if label==5:
        return np.maximum(lo-x,0)+np.maximum(x-hi,0)
    return np.maximum(0, (hi-lo)/2-np.abs(x-(lo+hi)/2))


def fit_curve(x, y, label):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if ymax-ymin < 1e-6:
        return None
    xx, yy = (x-xmin)/(xmax-xmin), (y-ymin)/(ymax-ymin)
    def linear(params):
        z=shape(xx,label,params)
        design=np.column_stack((z,np.ones(len(z))))
        coef=np.linalg.lstsq(design,yy,rcond=None)[0]
        loss=np.mean((design@coef-yy)**2)
        if coef[0]<=0 or (len(params)==2 and abs(params[0]-params[1])<.06):
            loss+=10
        return loss,coef
    n=1 if label in (0,1,4,7) else 2
    result=differential_evolution(lambda p:linear(p)[0],[(.08,.92)]*n,seed=731,popsize=10,maxiter=70,tol=1e-6)
    error,coef=linear(result.x)
    if error>.0025:
        return None
    params=sorted(result.x)
    kinks=params if label!=6 else [params[0],sum(params)/2,params[1]]
    def value(prices):
        return (coef[0]*shape((np.asarray(prices)-xmin)/(xmax-xmin),label,params)+coef[1])*(ymax-ymin)+ymin
    strikes=np.array(kinks)*(xmax-xmin)+xmin
    edges=np.unique(np.r_[0,strikes,xmax])
    values=value(edges)
    roots=[]
    for a,b,va,vb in zip(edges[:-1],edges[1:],values[:-1],values[1:]):
        if va*vb<0:
            roots.append(float(a-va*(b-a)/(vb-va)))
    right_slope=(value([xmax+1])[0]-value([xmax])[0])
    return dict(strikes=[round(float(k),2) for k in strikes], breakevens=[round(r,2) for r in roots], max_profit="unbounded" if right_slope>1e-5 else round(float(max(values)),2), max_loss="unbounded" if right_slope < -1e-5 else round(float(max(0,-min(values))),2), fit_error=float(error), observed_price_range=[round(float(xmin),2),round(float(xmax),2)])


def measure(image, label, ocr=None, vertical_axis="auto", axis_points=None):
    if vertical_axis not in ("auto", "profit_loss", "payoff"):
        raise ValueError("Unknown vertical-axis interpretation")
    confirmed = manual_axes([] if axis_points is None else axis_points)
    run_ocr = ocr is None
    ocr = ocr if ocr is not None else recognize(image)
    base=dict(available=False, message=ocr["message"], values=None)
    if not ocr["available"] and not (confirmed["x"] and confirmed["y"]):
        return base
    tokens=ocr["tokens"]
    text=" ".join(t["text"] for t in tokens).lower()
    if any(s in text for s in ("log scale", "logarithmic", "%")):
        return dict(base,message="Numerical extraction currently requires linear, non-percent axes.")
    xa,ya=calibrate(tokens)
    xa,ya=confirmed["x"] or xa,confirmed["y"] or ya
    if run_ocr and (xa is None or ya is None):
        secondary=recognize(image,engine="rapid")
        for token in secondary["tokens"]:
            value=number(token["text"])
            if value is None:
                continue
            cx=(token["box"][0]+token["box"][2])/2
            cy=(token["box"][1]+token["box"][3])/2
            if not any(number(t["text"])==value and abs((t["box"][0]+t["box"][2])/2-cx)<.025 and abs((t["box"][1]+t["box"][3])/2-cy)<.025 for t in tokens):
                tokens.append(token)
        automatic_x,automatic_y=calibrate(tokens)
        xa,ya=confirmed["x"] or automatic_x,confirmed["y"] or automatic_y
    if xa is not None and ya is None:
        # Recover horizontal kink positions without inventing a vertical scale.
        from .geometry_reader import candidates
        estimates=[]
        width=image.width*min(1,500/max(image.size))
        for xx,yy in candidates(image):
            fitted=fit_curve(xx/width*xa["slope"]+xa["intercept"],yy,label)
            if fitted is not None and fitted["fit_error"] < .0008:
                estimates.append(fitted)
        if estimates:
            values=min(estimates,key=lambda r:r["fit_error"])
            values.update(breakevens=[],max_profit=None,max_loss=None)
            return dict(available=True,values=values,axis_evidence=dict(x=xa,y=None),message="Estimated strike locations from the price axis and curve kinks, assuming the suggested strategy. The vertical scale was not readable; breakeven and profit/loss values are withheld.")
    if xa is None or ya is None:
        return dict(base,message="Could not verify the axis scales. At least two price ticks and three vertical ticks are needed for full numerical extraction.")
    # Independently colored expiration curves can be separated from a dashed
    # pre-expiration curve. Use full-image coordinates, not the tick-label row
    # as a plot boundary (labels may sit along the central zero axis).
    from .geometry_reader import candidates
    width, height = image.size
    scale = min(1, 500 / max(image.size))
    estimates=[]
    for xx, yy in candidates(image):
        fitted=fit_curve(xx/(width*scale)*xa["slope"]+xa["intercept"], -yy/(height*scale)*ya["slope"]+ya["intercept"], label)
        if fitted is not None and fitted["fit_error"] < .0008:
            estimates.append(fitted)
    if not estimates:
        curve=trace(image,xa,ya,tokens)
        if curve is not None:
            x,y=curve
            fitted=fit_curve(x*xa["slope"]+xa["intercept"], y*ya["slope"]+ya["intercept"],label)
            if fitted is not None:
                estimates.append(fitted)
    if not estimates:
        return dict(base,message="The traced curve does not reliably match the predicted shape; numerical values are withheld.")
    values=min(estimates,key=lambda r:r["fit_error"])
    price_span=values["observed_price_range"][1]-values["observed_price_range"][0]
    for candidate in estimates:
        if np.max(np.abs(np.array(candidate["strikes"])-values["strikes"])) > .025*price_span:
            return dict(base,message="More than one curve matches, with different strike locations. Crop to the expiration curve before reading values.")
    is_profit=vertical_axis=="profit_loss" or (vertical_axis=="auto" and any(s in text for s in ("profit", "p/l", "p&l", "p / l")))
    if not is_profit:
        values["max_profit"]=None
        values["max_loss"]=None
        values["breakevens"]=[]
    return dict(available=True, values=values, message="Approximate values from visible linear axes and a fitted curve. Strike locations assume the predicted strategy; amounts use the chart's units, with no contract multiplier. " + ("Profit/loss limits assume the full expiration curve and a nonnegative underlying price." if is_profit else "Profit/loss labeling was not verified, so breakevens and profit/loss limits are withheld."), axis_evidence=dict(x=xa,y=ya))
