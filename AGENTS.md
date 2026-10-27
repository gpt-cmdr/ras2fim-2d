# Agent rules for ras2fim-2d
Author and maintainer: Andy Carter, PE (Center for Water and the Environment, UT Austin).
These conventions are deliberate. Match them; do not "modernize" them.

## Pipeline
- Steps are scripts named <verb>_<noun>_NN.py; sub-steps use letters (02b), _mp_ marks multiprocessing.
- ras2fim-2d.py imports fn_<name>_NN from each step and runs a step range (-s "(start,end)").
- Steps communicate through numbered output folders (00_..., 01_...). Keep that layout.
- A new version of a step is a new file (_v2, _v3). Do not edit the old version in place unless asked.
- Every step must still run standalone through its own __main__ / argparse block.

## Code style
- File header: project, script number, "Created by: Andy Carter, PE", Created - YYYY.MM.DD.
  When you change a file, add: "# Revised - YYYY.MM.DD -- <what changed>".
- Names: str_, int_, flt_, b_, list_, dict_, tup_, gdf_, df_, ds_, arr_ prefixes; functions fn_snake_case.
- Frame each function with a comment rule line, as the surrounding file does.
- Procedural functions. No classes, type hints, dataclasses, or logging module unless asked.
- Output with print: keep the banner, "---(i) LABEL:" argument lines, the tqdm bar_format already
  used in the file, and the final "Compute Time:" line.
- argparse: short flags, dest names with prefixes, help strings starting REQUIRED: / OPTIONAL: ... Default:.

## Configs
- INI via configparser, one section per step ([02_copy_geom]). Several environment variants exist
  (config_global.ini, config_global_windows.ini, ...). A key change goes into every variant.
- Keep commented-out old values. Preserve each file's existing line endings.

## Do not
- Rename existing keys, functions, files, or output names, even misspelled ones; they are interfaces.
- Reformat, reorder imports, or tidy code you were not asked to change.
- Add dependencies, change the conda env / Python version, or alter the Dockerfile without asking.
- Remove "Created by" credit or the CWE/UT banner.
- Commit or push. Propose the change and a short commit message; Andy commits.

## Checking work
- There is no unit test suite. Verify by running the sample data and inspecting outputs.
  Say exactly what you ran and what you saw; if you could not run it, say so.

## Linux/Wine preprocessing container (contributed by CLB Engineering)
- containers/hecras-prepare/ builds the optional image that runs step 02b HEC-RAS preprocessing on a
  Linux Docker host. src/stage_hecras_for_linux_wine_02b.py drives it. Docs: containers/hecras-prepare/README.md.
- It is used only when [03_run_hec_ras] b_prepare_on_linux_wine = True. The default path,
  stage_hecras_for_linux_v66_02b.py on the Windows machine, must keep working unchanged.
- The container code is CLB-maintained and written in a different style. Do not restyle it, and do not
  copy that style into src/.
- tests/ covers only this container path (python -m pytest tests). Keep HEC-RAS installers, Wine
  profiles, model data and results out of Git.
