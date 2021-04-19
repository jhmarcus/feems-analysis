#!python

include: "snakefiles/run_eems/Snakefile"
include: "snakefiles/run_admixture/Snakefile"
include: "snakefiles/run_simulations/Snakefile"

rule none:
    input: "Snakefile"
    run: print("feems-analysis")
