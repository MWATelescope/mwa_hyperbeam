// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use criterion::*;

use hyperbeam::*;

// This benchmark relies on `Hyperbeam::calc_modes` being made public. It should
// normally be left as private.
fn coefficients(c: &mut Criterion) {
    let mut beam = Hyperbeam::new("mwa_full_embedded_element_pattern.h5").unwrap();

    c.bench_function("calculating coefficients", |b| {
        let freq = 51200000;
        let delays = [0; 16];
        let gains = [1.0; 16];
        b.iter(|| {
            // By calling calc_modes, we skip the cache.
            let result = beam.calc_modes(freq, &delays, &gains).unwrap();
        })
    });

    c.bench_function("getting coefficients from cache", |b| {
        let freq = 51200000;
        let delays = [0; 16];
        let gains = [1.0; 16];
        // Prime the cache.
        let result = beam.get_modes(freq, &delays, &gains).unwrap();
        b.iter(|| {
            // By calling get_modes, we hit the cache.
            let result = beam.get_modes(freq, &delays, &gains).unwrap();
        })
    });
}

criterion_group!(benches, coefficients);
criterion_main!(benches);
