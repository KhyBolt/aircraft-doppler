import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from . import __version__
from .audio import read_audio, make_spectrogram, save_spectrogram, load_spectrogram
from .io import load_inputs, load_config, write_geographic_grid
from .search import search, best_per_pair


def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Experimental Doppler observer localization. Accuracy unvalidated on field recordings.")
    sub = parser.add_subparsers(dest="command",required=True)
    demo = sub.add_parser("demo",help="Run an independent synthetic two-flyover benchmark")
    demo.add_argument("--output",default="validation/demo")
    bench = sub.add_parser("benchmark", help="Replay bundled offline synthetic WAV without event picks")
    bench.add_argument("--fixture", default="fixtures")
    bench.add_argument("--output", required=True)
    wide = sub.add_parser("benchmark-wide", help="Run the fixed 27-minute, 10 × 10 km synthetic suite")
    wide.add_argument("--output", required=True)
    wide.add_argument("--profile", choices=("guarded","baseline"), default="guarded",
                      help="Use v0.4 guarded mode (default) or replay the v0.3 search gates.")
    from .wide_fixture import SCENES
    wide.add_argument("--case", action="append", choices=[scene.slug for scene in SCENES],
                      help="Only run the selected synthetic case; repeat to select more than one.")
    wide.add_argument("--max-seconds", type=int, default=540, choices=range(1,541), metavar="1..540",
                      help="Cooperative wall-time budget (default 540; case-by-case checkpoint).")
    gen = sub.add_parser("make-fixture", help="Regenerate the from-scratch synthetic fixture")
    gen.add_argument("--output", required=True)
    fit = sub.add_parser("fit-trace",help="Fit an ideal straight-flight Doppler curve to a traced fundamental")
    fit.add_argument("trace",help="CSV columns audio_time_s,frequency_hz")
    fit.add_argument("--initial",nargs=4,type=float,required=True,metavar=("SPEED","SLANT","F0","DELAYED_CPA"))
    fit.add_argument("--sound-speed",type=float,default=343)
    fit.add_argument("--output",required=True)
    sp = sub.add_parser("spectrogram",help="Make a cached STFT from audio/video")
    sp.add_argument("audio")
    sp.add_argument("output")
    sp.add_argument("--rate",type=int,default=2000)
    sp.add_argument("--channel",type=int,default=0)
    sp.add_argument("--window",type=float,default=2)
    sp.add_argument("--hop",type=float,default=0.25)
    sp.add_argument("--fmin",type=float,default=40)
    sp.add_argument("--fmax",type=float,default=700)
    run = sub.add_parser("search",help="Search supplied bounded candidates and normalized tracks")
    run.add_argument("--tracks",required=True)
    run.add_argument("--candidates",required=True)
    run.add_argument("--config",required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio")
    source.add_argument("--spectrogram")
    run.add_argument("--output",required=True)
    run.add_argument("--channel",type=int,default=0)
    grid = sub.add_parser("terrain-grid",help="Local DEM + geoid candidate grid (optional geospatial dependencies)")
    grid.add_argument("--bounds",type=float,nargs=4,required=True,metavar=("WEST","SOUTH","EAST","NORTH"))
    grid.add_argument("--spacing",type=float,default=200)
    grid.add_argument("--dem",required=True)
    grid.add_argument("--geoid",required=True)
    grid.add_argument("--output",required=True)
    grid.add_argument("--mic-height",type=float,default=1.5)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run_demo
        print(json.dumps(run_demo(args.output),indent=2))
    elif args.command == "benchmark":
        from .benchmark import benchmark
        result = benchmark(args.fixture, args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "supported_in_this_fixture" else 2
    elif args.command == "benchmark-wide":
        from .wide import run
        info = run(args.output,case_slugs=args.case,max_seconds=args.max_seconds,
                   hardened=args.profile=="guarded",progress=lambda msg: print(msg,flush=True))
        cases = info["cases_completed"]
        print(json.dumps({"status":info["status"],"verified_cases":len(cases),"completed_case_names":list(cases),
                          "matching_expectation":sum(x["expected_behavior_observed"] for x in cases.values()),
                          "elapsed_wall_s":info["elapsed_wall_s"]},indent=2))
        return 3 if info["status"] != "completed" else (0 if all(x["expected_behavior_observed"] for x in cases.values()) else 2)
    elif args.command == "make-fixture":
        from .benchmark import make_fixture
        make_fixture(args.output)
    elif args.command == "fit-trace":
        import pandas as pd
        from .measurement import fit_ideal_trace
        table = pd.read_csv(args.trace)
        result = fit_ideal_trace(table.audio_time_s,table.frequency_hz,args.initial,args.sound_speed)
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    elif args.command == "spectrogram":
        sr,x = read_audio(args.audio,args.rate,args.channel)
        spec = make_spectrogram(x,sr,args.window,args.hop,args.fmin,args.fmax)
        save_spectrogram(spec,args.output)
        print(f"Saved {len(spec.t)} time frames and {len(spec.f)} frequency bins")
    elif args.command == "terrain-grid":
        print(write_geographic_grid(args.bounds,args.spacing,args.dem,args.geoid,args.output,args.mic_height))
    elif args.command == "search":
        out = Path(args.output)
        out.mkdir(parents=True,exist_ok=True)
        events,settings,stft_settings = load_config(args.config)
        if args.spectrogram:
            spec = load_spectrogram(args.spectrogram)
        else:
            sr,x = read_audio(args.audio,channel=args.channel)
            spec = make_spectrogram(x,sr,**stft_settings)
        tracks, candidates, frame = load_inputs(args.tracks, args.candidates)
        scores,audit = search(tracks,candidates,events,spec,settings,lambda s:print(s,file=sys.stderr))
        scores.to_csv(out/"scores.csv",index=False)
        best_per_pair(scores).to_csv(out/"best_per_pair.csv",index=False)
        inputs = {k:v for k,v in vars(args).items() if k in ("tracks","candidates","config","audio","spectrogram") and v}
        manifest = {"method":f"experimental v{__version__}; homogeneous still-air propagation", "accuracy":"not field validated",
                    "inputs":{k:{"path":str(v),"sha256":sha256(v)} for k,v in inputs.items()},
                    "events":[asdict(e) for e in events],"search":asdict(settings),"stft":stft_settings,
                    "coordinate_frame":frame,"channel":args.channel,"audit":audit,"cached_stft_settings_verified":False if args.spectrogram else True,
                    "python":sys.version,"numpy":np.__version__}
        (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
        print(json.dumps({"matches":len(scores),"output":str(out)},indent=2))
        return 0 if len(scores) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
