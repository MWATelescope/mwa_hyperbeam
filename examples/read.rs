// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*!
Test reading the hdf5 file containing the spherical harmonics.
 */

use hyperbeam::*;
use structopt::*;

#[derive(StructOpt, Debug)]
struct Opt {
    /// Path to the HDF5 file.
    #[structopt(parse(from_os_str))]
    hdf5_file: std::path::PathBuf,
}

fn main() {
    let opts = Opt::from_args();
    // let d = hyperbeam::MwaH5Data::new(&opts.hdf5_file).unwrap();
    // println!("{:?}", d);
}
