# Comparisons between a few implementations of the MWA FEE beam code

## Performance summary

| Package | Config | Number of directions | Duration | Max. memory usage |
|:--------|:-------|---------------------:|---------:|------------------:|
| [mwa_pb](https://github.com/MWATelescope/mwa_pb) | serial           | 1       | 14.85 ms | 153 MiB  |
|                                                  | serial           | 1,000   | 130.1 ms | 201 MiB  |
|                                                  | serial           | 300,000 | 37.94 s  | 14.4 GiB |
| [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata) | serial | 32,760 | 7.446 s | 653 MiB |
|                                                  | serial           | 130,320 | 11.85 s  | 1.92 GiB |
| [EveryBeam](https://git.astron.nl/RD/EveryBeam)  | serial           | 1       | 114 µs   | 61.7 MiB |
|                                                  | serial           | 1,000   | 103.9 ms | 61.9 MiB |
|                                                  | serial           | 300,000 | 31.16 s  | 71.1 MiB |
| mwa_hyperbeam                                    | serial           | 1       | 32.54 µs | 11.1 MiB |
|                                                  | serial           | 1,000   | 29.02 ms | 13.3 MiB |
|                                                  | parallel         | 1,000   | 4.598 ms | 13.5 MiB |
|                                                  | serial           | 300,000 | 8.610 s  | 33.9 MiB |
|                                                  | parallel         | 300,000 | 596.1 ms | 34.6 MiB |
|                                                  | serial           | 999,999 | 28.66 s  | 87.3 MiB |
|                                                  | parallel         | 999,999 | 1.986 s  | 90.3 MiB |
|                                                  | CUDA             | 300,000 | 63.70 ms | 134 MiB  |
|                                                  | CUDA             | 999,999 | 164.4 ms | 195 MiB  |
|                                                  | serial, Python   | 1       | 36.95 µs | 42.5 MiB |
|                                                  | serial, Python   | 1,000   | 31.57 ms | 44.9 MiB |
|                                                  | parallel, Python | 1,000   | 3.890 ms | 45.5 MiB |
|                                                  | serial, Python   | 300,000 | 9.393 s  | 93.5 MiB |
|                                                  | parallel, Python | 300,000 | 660.3 ms | 96.3 MiB |
|                                                  | serial, Python   | 999,999 | 31.05 s  | 211 MiB  |
|                                                  | parallel, Python | 999,999 | 2.191 s  | 212 MiB  |
|                                                  | CUDA, Python     | 999,999 | 1.342 s  | 305 MiB  |

All of the durations refer to "hot cache" times. See the full print out of the
benchmarks run and the system details at the bottom of this page. You can verify
these numbers by running `run.sh`.

## Compared packages and their features

Please file an issue if this information is incorrect.

|                                           | mwa_pb             | pyuvdata           | EveryBeam | mwa_hyperbeam      |
|-------------------------------------------|:------------------:|:------------------:|:---------:|:------------------:|
| Can be run in parallel?                   | :x:                | :x:                | :x:       | :white_check_mark: |
| Parallactic-angle correction?             | :x:                | :x:                | :x:       | :white_check_mark: |
| GPU (CUDA/HIP) support?                   | :x:                | :x:                | :x:       | :white_check_mark: |
| Supports MWA analytic beam?               | :white_check_mark: | :x:                | :x:       | :white_check_mark: |
| Supports per-dipole gains?                | :white_check_mark: | :white_check_mark: | :x:       | :white_check_mark: |
| Python interface?                         | :white_check_mark: | :white_check_mark: | :x:*      | :white_check_mark: |
| Can be called from other languages via C? | :x:                | :x:                | :x:       | :white_check_mark: |
| Supports MWA CRAM tile?                   | :x:                | :x:                | :x:       | :white_check_mark: |

*: `EveryBeam` has a Python interface, but it does not support the MWA beam.

## Compared packages and methodology

To my knowledge, `EveryBeam` only takes RADec coordinates to do its simulations,
whereas others use azimuth and elevation. Also unlike others, `EveryBeam`
needs a measurement set for its work. To keep things as fair as possible, I've
provided a stripped-down measurement set and given it to `EveryBeam` with RADec
(0, -27). This corresponds to a specific AzEl which I've then used in other
packages.

### mwa_pb

Installed with `pip install mwa_pb`.

The latest pip-installable version needs a hack to work correctly with the
latest `numpy`. Change all instances of `numpy.complex` and `numpy.complex64` to
just `complex`. Also this line

```python
if not interp and "mwa_hyperbeam" in sys.modules:
```

needs to be changed to

```python
if False:
```

to actually use the pure-Python `mwa_pb`, otherwise `hyperbeam` is used
internally :)

Finally, the results here are a little misleading. All other examples use the
same azimuth and elevation for their simulations, and therefore the resulting
Jones matrices are all the same. This is fine, as all other code doesn't check
the input to de-duplicate (it's the caller's responsibility). However, `mwa_pb`
does notice this duplication. To actually get it to do some work, it is called
with different azimuths and elevations to other packages, which is a little
unfair.

In addition, `mwa_pb` has a nice interpolation feature, which reduces
the number of FEE simulations to be made by gridding coarsely and using "nearby"
simulations instead. Using interpolation makes the code quite fast, but would
also be a very unfair comparison, as other packages don't have interpolation and
using it compromises the accuracy of the results.

### EveryBeam

Version 0.5.3, alongside `casacore` 3.5.0. I had to manually install the MWA
header files, as the default `CMake` build doesn't seem to. I also used the
Python interface via `pip` (version 0.5.1) and from source (version 0.5.3);
neither supports MWA beam responses.

`EveryBeam` does not appear to have a way to do FEE calculations in parallel.
Attempting to compile with `OpenMP` and annotating a pragma on a for loop either
caused segfaults or showed HDF5 errors. Thus, `EveryBeam` is only using a single
thread in this comparison. If there is a way to make the code run in parallel,
please share it!

It also seems that the beam responses are normalised regardless of the beam
normalisation setting.

### pyuvdata

I used version 2.4.1 installed with `pip`.

`pyuvdata` includes a module `uvbeam`, which has MWA beam code. I'm not sure how
to use this code best in a fair comparison; it seems designed to generate beam
responses over a grid with adjustable resolution rather than allowing arbitrary
directions. I've opted to adjust the resolution and report the timings and
memory usage.

### mwa_hyperbeam

Also referred to as `hyperbeam`. Installed from source, and the Python interface
was also from source or via `pip`.

#### CUDA

`hyperbeam` can use CUDA-capable (or HIP-capable) GPUs to run the simulations
faster. Here, we are using the `gpu-single` feature; this means the simulations
are calculated with 32-bit floats, which benefits my desktop-grade GPU. Omitting
this feature will make the calculations use 64-bit floats, which run much
slower on my GPU. However, these floats will be crunched much faster on a
"datacenter"-grade GPU, such as those hosted by Pawsey.

## Omitted

### "Marcin's C++ code"

This code can be seen on
[this commit of `hyperbeam`](https://github.com/MWATelescope/mwa_hyperbeam/commit/72e0914).
From memory, this code isn't ready to be used as a library, so it's not easy
to do comparisons with. However, my (CHJ) testing from years ago suggested that
this code is 10x slower and uses much more memory than `hyperbeam`, while giving
extremely similar if not the same results.

### RTS FEE

The Real-Time System (RTS) has an implementation of the FEE code, but only for
CUDA. The RTS is not open sourced, so it is difficult to fairly compare this
code. However, Jack Line's testing showed that its results were consistent with
`hyperbeam` but approximately 3-4x slower.

## Benchmark print out

Obtained by running `run.sh`.

```
*** System information ***
uname -a:
  Linux sirius 6.5.2-arch1-1 #1 SMP PREEMPT_DYNAMIC Wed, 06 Sep 2023 21:01:01 +0000 x86_64 GNU/Linux
CPU:
  AMD Ryzen 9 3900X 12-Core Processor (12 cores, 24 threads)
GPU:
  NVIDIA GeForce RTX 2070
Total memory:
  128735 MiB
glibc:
  GNU C Library (GNU libc) stable release version 2.38.
Compilers:
  GCC: g++ (GCC) 13.2.1 20230801
  Rust: rustc 1.75.0 (82e1608df 2023-12-21)
  nvcc: Cuda compilation tools, release 12.2, V12.2.91
Python:
  Python 3.11.5
***

*** mwa_pb Python results ***
time taken to produce 1 simulation (cold cache): 0.04600405693054199s
time taken to produce 1 simulation (hot cache):  0.01485133171081543s
time taken to produce 1000 simulations (hot cache):  0.13009285926818848s
time taken to produce 300000 simulations (hot cache):  37.93662881851196s
Max memory use (kBytes): 15080124
***

*** pyuvdata Python results ***
time taken to produce 32760 simulations: 7.445570707321167s
time taken to produce 130320 simulations: 11.852437734603882s
Max memory use (kBytes): 2459428
***

*** hyperbeam Python results with 1 CPU core ***
time taken to produce 1 simulation (cold cache): 0.001977205276489258s
time taken to produce 1 simulation (hot cache):  3.695487976074219e-05s
time taken to produce 1000 simulations (hot cache):  0.031566619873046875s
time taken to produce 300000 simulations (hot cache):  9.393456220626831s
time taken to produce 999999 simulations (hot cache):  31.051132917404175s
First and last MWA beam responses:
[ 0.95149467+0.23737095j -0.16821772-0.04190302j -0.1687059 -0.04202984j
 -0.95181881-0.23673898j]
[ 0.95149467+0.23737095j -0.16821772-0.04190302j -0.1687059 -0.04202984j
 -0.95181881-0.23673898j]
Max memory use (kBytes): 235028

*** hyperbeam Python results with all CPU cores ***
time taken to produce 1 simulation (cold cache): 0.001982450485229492s
time taken to produce 1 simulation (hot cache):  3.6716461181640625e-05s
time taken to produce 1000 simulations (hot cache):  0.0038902759552001953s
time taken to produce 300000 simulations (hot cache):  0.6602745056152344s
time taken to produce 999999 simulations (hot cache):  2.190699577331543s
First and last MWA beam responses:
[ 0.95149467+0.23737095j -0.16821772-0.04190302j -0.1687059 -0.04202984j
 -0.95181881-0.23673898j]
[ 0.95149467+0.23737095j -0.16821772-0.04190302j -0.1687059 -0.04202984j
 -0.95181881-0.23673898j]

*** hyperbeam Python results with CUDA ***
time taken to produce 999999 simulations (hot cache):  1.3418140411376953s
Max memory use (kBytes): 381324
***

*** Compiling EveryBeam C++ example ***
make: Nothing to be done for 'all'.

*** EveryBeam C++ results with 1 CPU core ***
time taken to produce 1 simulation (cold cache): 0.099906s
time taken to produce 1 simulation (hot cache):  0.000114s
time taken to produce 1000 simulations: 0.103940s
time taken to produce 300000 simulations: 31.161970s

First and last MWA beam responses:
[+0.951391+0.237339i, -0.168200-0.041897i
 -0.168688-0.042025i, -0.951721-0.236713i]
[+0.951391+0.237339i, -0.168200-0.041897i
 -0.168688-0.042025i, -0.951721-0.236713i]
Max memory use (kBytes): 72784
***

*** Compiling hyperbeam Rust code ***

*** hyperbeam Rust results with 1 CPU core ***
time taken to produce 1 simulation (cold cache): 1.936581ms
time taken to produce 1 simulation (hot cache): 32.542µs
time taken to produce 1000 simulations (hot cache): 29.02333ms
time taken to produce 300000 simulations (hot cache): 8.609547797s
time taken to produce 999999 simulations (hot cache): 28.656200548s
First and last MWA beam responses:
[+0.951495+0.237371i, -0.168218-0.041903i
 -0.168706-0.042030i, -0.951819-0.236739i]
[+0.951495+0.237371i, -0.168218-0.041903i
 -0.168706-0.042030i, -0.951819-0.236739i]
Max memory use (kBytes): 91416

*** hyperbeam Rust results with all CPU cores ***
time taken to produce 1 simulation (cold cache): 1.884332ms
time taken to produce 1 simulation (hot cache): 30.046µs
time taken to produce 1000 simulations (hot cache): 4.598173ms
time taken to produce 300000 simulations (hot cache): 596.130603ms
time taken to produce 999999 simulations (hot cache): 1.985971281s
First and last MWA beam responses:
[+0.951495+0.237371i, -0.168218-0.041903i
 -0.168706-0.042030i, -0.951819-0.236739i]
[+0.951495+0.237371i, -0.168218-0.041903i
 -0.168706-0.042030i, -0.951819-0.236739i]
Max memory use (kBytes): 92028

*** hyperbeam Rust results with CUDA ***
time taken to produce 300000 simulations (cold cache): 1.069158203s
time taken to produce 300000 simulations (hot cache):  63.699647ms
time taken to produce 999999 simulations (hot cache):  164.427598ms
First and last MWA beam responses:
[+0.951469+0.237365i, -0.168213-0.041902i
 -0.168701-0.042029i, -0.951793-0.236733i]
[+0.951469+0.237365i, -0.168213-0.041902i
 -0.168701-0.042029i, -0.951793-0.236733i]
Max memory use (kBytes): 205064
***
```

