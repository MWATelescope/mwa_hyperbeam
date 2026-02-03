use std::f64::consts::FRAC_PI_2;

use marlu::AzEl;

/// A trait that describes a coordinate pair in a horizonal coordinate system.
/// MWA beam codes historially specified (azimuth, zenith angle) rather than
/// the perhaps-more-familiar "alt az" (altitude, azimuth), so we conform with
/// history here. The MWA also prefers "elevation" instead of "altitude".
pub trait HorizCoord: Copy {
    /// Get the azimuth of this coordinate.
    fn get_az(&self) -> f64;
    /// Get the elevation of this coordinate.
    fn get_el(&self) -> f64;
    /// Get the zenith angle of this coordinate.
    fn get_za(&self) -> f64;
}

impl<C: HorizCoord> HorizCoord for &C {
    fn get_az(&self) -> f64 {
        (*self).get_az()
    }

    fn get_el(&self) -> f64 {
        (*self).get_el()
    }

    fn get_za(&self) -> f64 {
        (*self).get_za()
    }
}

impl HorizCoord for AzEl {
    fn get_az(&self) -> f64 {
        self.az
    }

    fn get_el(&self) -> f64 {
        self.el
    }

    fn get_za(&self) -> f64 {
        self.za()
    }
}

/// We assume that a tuple of floats is (azimuth, zenith angle), both in
/// radians.
impl HorizCoord for (f64, f64) {
    fn get_az(&self) -> f64 {
        self.0
    }

    fn get_el(&self) -> f64 {
        FRAC_PI_2 - self.1
    }

    fn get_za(&self) -> f64 {
        self.1
    }
}

impl HorizCoord for (&f64, &f64) {
    fn get_az(&self) -> f64 {
        *self.0
    }

    fn get_el(&self) -> f64 {
        FRAC_PI_2 - self.1
    }

    fn get_za(&self) -> f64 {
        *self.1
    }
}
