"""Report native benchmark outputs; native RMSE implies R2 on the same test target."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tt_curation.criteria import curation_verdict, evaluation_report


def report(output):
    from multabench.benchmark.load import load_split
    from multabench.datasets.linkedin_salary import FULL, STRUCTURED, TEXT
    output=Path(output)
    rows=[]
    for path in sorted((output/"runs").glob("*.json")):
        r=json.loads(path.read_text(encoding="utf-8"))
        if r.get("status") != "ok": raise ValueError("Failed native run: "+str(path))
        dataset=r["dataset"]
        condition={STRUCTURED.name:"structured",TEXT.name:"unstructured"}.get(dataset)
        if dataset==FULL.name:
            condition={"e5-small":"joint_frozen","e5-small-tar":"joint_tar"}.get(r.get("text_encoder"))
        if condition is None or r["metric"]!="root_mean_squared_error" or r["n_train"]!=10000 or r["n_test"]!=5000:
            raise ValueError("Unexpected native result identity: "+str(path))
        split=load_split(FULL,r["fold"],size="10k")
        variance=float(np.var(split.y_test.to_numpy(),ddof=0))
        # Native regression target preprocessing is identity; y is raw log_salary.
        r2=1-float(r["test_error"])**2/variance
        rows.append({**r,"condition":condition,"r2":r2})
    frame=pd.DataFrame(rows)
    if frame.empty: raise ValueError("No native results")
    if frame.duplicated(["model","condition","fold"]).any(): raise ValueError("Duplicate native evaluations")
    complete=[]
    models={"TabM","CatBoost","LightGBM","TabPFN-v2","TabPFN-v2p5"}
    conditions={"structured","unstructured","joint_frozen","joint_tar"}
    for fold,group in frame.groupby("fold"):
        if set(zip(group.model,group.condition)) == {(m,c) for m in models for c in conditions}: complete.append(int(fold))
    frame.to_csv(output/"per_fold_results.csv",index=False)
    subset=frame[frame.fold.isin(complete)]
    if subset.empty: print(f"INCOMPLETE: {len(frame)}/120, no full fold"); return False
    evaluation_report(subset).to_csv(output/"evaluation_report.csv",index=False)
    verdict=curation_verdict(subset.assign(test_score=subset.r2))
    final=len(frame)==120 and sorted(complete)==list(range(6))
    verdict.to_csv(output/("curation_verdict.csv" if final else "preliminary_verdict.csv"),index=False)
    if not final: (output/"curation_verdict.csv").unlink(missing_ok=True)
    text=f"{'FINAL' if final else 'PRELIMINARY'} native evaluation: {len(frame)}/120; complete folds {complete}\n"
    text+=verdict.to_string(index=False,float_format=lambda x:f"{x:.6f}")
    text+=f"\nJoint Signal: {int(verdict.joint_signal.sum())}/5; TAR Gain: {int(verdict.tar_gain.sum())}/5 (independent).\n"
    (output/"summary.txt").write_text(text,encoding="utf-8"); print(text)
    return final


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--results",required=True)
    report(p.parse_args().results)
