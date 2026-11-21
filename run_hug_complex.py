"""Blind-well classification of the framework network with the complex (Re/Im) CWT representation selected by
leave-one-well-out tuning on the training wells. Usage: python3 -I run_hug_complex.py <repo> <n_seeds> <out_json>"""
import sys, runpy
sys.path.insert(0, ".")
import core
core.CWT_MODE = "complex"
repo, nseed, out = sys.argv[1:4]
sys.argv = ["hugoton.py", repo, "blind", nseed, out, "hyb_bigru"]
runpy.run_path("hugoton.py", run_name="__main__")
