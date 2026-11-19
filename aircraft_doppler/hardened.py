"""Same-input A/B replay of the baseline and guarded v0.4 profile."""
from pathlib import Path
import json
import sys
from time import perf_counter
from .benchmark import _write_json
from .wide import BudgetExceeded, _deadline_check, infer, evaluate
from .wide_fixture import Scene, make_scene, SCENES

# Every seeded scene and threshold is fixed before the benchmark is scored.
FRESH = (
    Scene("fresh-southeast", (4489.0,-4391.0), -66.625, 2987, 343.2, "positive","fresh",5783),
    Scene("fresh-southwest", (-3871.0,-4557.0), 57.375, 870, 342.6, "positive","fresh",8804),
    Scene("fresh-northeast", (4078.0,3339.0), 69.125, 1916, 343.55, "positive","fresh",631),
    Scene("fresh-center", (-687.0,971.0), -26.625, 5374, 343.3, "positive","fresh",8833),
    Scene("missing-fresh", (447.0,-1785.0), -35.625, 7884, 343.15, "wrong_tracks","fresh",973),
    Scene("decoy-near", (-1159.0,2365.0), 39.375, 5656, 342.8, "near_decoy","fresh",670),
    Scene("shifted-clock", (4151.0,-2388.0), -40.875, 9740, 343.4, "shifted_tracks","fresh",918),
    Scene("ambiguous-fresh", (-2089.0,1728.0), -12.875, 6253, 343.0, "ambiguous","fresh",34),
    Scene("exact-alias", (-1976.0,1638.0), -14.625, 5308, 343.0, "exact_alias","fresh",6071),
    Scene("no-flyover-fresh", (-873.0,618.0), -54.875, 1727, 343.0, "no_flyovers","fresh",89),
)
REGRESSIONS = tuple(scene for scene in SCENES if scene.split != "pilot")
EXACT_LIMIT = Scene("undetectable-alias", (-1937.0,1743.0), -27.750, 6527, 343.0, "exact_alias","edge-case")
COMPARE_SCENES = (next(s for s in REGRESSIONS if s.slug=="wrong-tracks"), *FRESH,
                  *(s for s in REGRESSIONS if s.slug!="wrong-tracks"),EXACT_LIMIT)


def run(output,*,case_slugs=None,max_seconds=540,progress=print):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    start = perf_counter()
    deadline = start + max_seconds
    scenes = [scene for scene in COMPARE_SCENES if
              (scene.slug in case_slugs if case_slugs else (scene.split=="fresh" or scene.slug=="wrong-tracks"))]
    info = {"protocol":"paired same-input synthetic comparison","status":"running","max_seconds":max_seconds,
            "planned_cases":[scene.slug for scene in scenes],"cases":{}}
    def checkpoint():
        info["metrics"] = {}
        for version in ("v0.3","v0.4"):
            results = [case[version] for case in info["cases"].values()]
            positives = [result for result in results if result["truth"]["expected_outcome"]=="point"]
            controls = [result for result in results if result["truth"]["expected_outcome"]=="abstain"]
            info["metrics"][version] = {
                "positive_cases":len(positives),
                "correct_accepted_positives":sum(result["expected_behavior_observed"] for result in positives),
                "all_accepted_positives":sum("estimated_xyz_m" in result for result in positives),
                "controls":len(controls),
                "false_accepts":sum("estimated_xyz_m" in result for result in controls),
                "inference_wall_s_total":round(sum(result["inference_wall_s"] for result in results),3),
            }
        info["wall_s"] = round(perf_counter()-start,3)
        temporary = output/"comparison.tmp"
        _write_json(temporary,info)
        temporary.replace(output/"comparison.json")
    checkpoint()
    for scene in scenes:
        try:
            _deadline_check(deadline)
            progress(f"{scene.slug}: generating identical inputs for both profiles")
            folder = output/scene.slug
            inputs = folder/"inputs"
            make_scene(inputs,scene)
            args = [inputs/name for name in ("synthetic.wav","tracks.csv","candidates.csv")]
            scored = {}
            for name,enable in (("v0.3",False),("v0.4",True)):
                _deadline_check(deadline)
                blind = infer(*args, folder/name, deadline=deadline, hardened=enable)
                # No fixture answer is read by either estimator.
                truth = json.loads((inputs/"truth.json").read_text(encoding="utf-8"))
                result = evaluate(blind,truth)
                _write_json(folder/name/"summary.json",result)
                scored[name] = result
                progress(f"{scene.slug} {name}: {result['status']} in {result['inference_wall_s']:.1f} s")
            assert scored["v0.3"]["inputs_sha256"] == scored["v0.4"]["inputs_sha256"]
            info["cases"][scene.slug] = scored
            checkpoint()
        except BudgetExceeded:
            info["status"] = "timed_out_partial"
            info["first_unfinished_case"] = scene.slug
            break
    else:
        info["status"] = "completed"
    checkpoint()
    return info


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare guarded v0.4 against v0.3 on generated audio only.")
    parser.add_argument("--output",required=True)
    parser.add_argument("--case",action="append",choices=[scene.slug for scene in COMPARE_SCENES])
    parser.add_argument("--max-seconds",type=int,default=540,choices=range(1,541),metavar="1..540")
    ns = parser.parse_args()
    result = run(ns.output,case_slugs=ns.case,max_seconds=ns.max_seconds,progress=lambda s:print(s,flush=True))
    print(json.dumps({"status":result["status"],"completed_cases":len(result["cases"]),
                      "wall_s":result["wall_s"],"metrics":result["metrics"]},indent=2))
    return 0 if result["status"] == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
