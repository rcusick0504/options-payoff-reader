"""Reproduce the fixed 24-chart numerical development audit with local OCR."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .diverse import render_diverse
from .measure import measure
from .payoffs import construct


def run():
    expected = [[100],[100],[90,110],[90,110],[100],[90,110],[80,100,120],[100]]
    rows=[]
    errors=[]
    for family in [2,12,22]:
        for label in range(8):
            trade=dict(legs=[asdict(leg) for leg in construct(label)],xlim=[40,160])
            image=render_diverse(trade,family,81+label,(800,600))
            result=measure(image,label)
            rows.append(dict(family=family,label=label,expected_strikes=expected[label],result=result))
            if result["available"]:
                errors.extend(np.abs(np.array(result["values"]["strikes"])-expected[label]).tolist())
    payload=dict(purpose="Numerical development audit with known category supplied; no manual tick corrections",n=len(rows),recovered=sum(r["result"]["available"] for r in rows),median_strike_error=float(np.median(errors)) if errors else None,max_strike_error=max(errors) if errors else None,rows=rows)
    Path("docs/results/numerical_v2_audit.json").write_text(json.dumps(payload,indent=2)+"\n")
    print({k:v for k,v in payload.items() if k!="rows"})


if __name__=="__main__":
    run()
