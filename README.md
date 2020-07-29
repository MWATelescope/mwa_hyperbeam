# mwa_hyperbeam

Primary beam code for the Murchison Widefield Array (MWA) radio telescope.

This code exists to provide a single correct, convenient implementation of
[Marcin
Sokolowski's](https://ui.adsabs.harvard.edu/abs/2017PASA...34...62S/abstract)
Full Embedded Element (FEE) primary beam model of the MWA, a.k.a. "the 2016
beam". This code should be used over all others. If there are soundness issues,
please raise them here so everyone can benefit.

## Usage
See Rust and C examples in the `examples` directory.

## Installation

### Prerequisites
<details>

- Cargo and a Rust compiler. `rustup` is recommended:

  `https://www.rust-lang.org/tools/install`

- [hdf5](https://www.hdfgroup.org/hdf5)

</details>

### From source

Clone the repo, and run:

    cargo build --release

For usage with other languages, an include file will be in the `include`
directory, along with shared and static objects in the `target/release`
directory.

## Troubleshooting

Run your code with `hyperbeam` again, but this time with the debug build. This should be as simple as running:

    cargo build
    
And then linking against the object in `./target/debug`.

If that doesn't help reveal the problem, report the version of the software
used, your usage and the program output in a new GitHub issue.

## hyperbeam?
AERODACTYL used HYPER BEAM!
