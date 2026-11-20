/* Rust bindings to src/ska/gpu/ska.h, in the style of rust-bindgen (see
 * src/update_rust_bindings.sh). */

pub const SKA_MODEL_SKA_ANALYTIC: SKA_MODEL = 0;
pub const SKA_MODEL_SKA_EEP_GRID: SKA_MODEL = 1;
pub const SKA_MODEL_SKA_EEP_SPH: SKA_MODEL = 2;
pub type SKA_MODEL = ::std::os::raw::c_uint;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SkaGpuModel {
    pub model: SKA_MODEL,
    pub cos_taper: ::std::os::raw::c_int,
    pub phis: *const f64,
    pub num_phis: ::std::os::raw::c_int,
    pub thetas: *const f64,
    pub num_thetas: ::std::os::raw::c_int,
    pub planes: *const ::std::os::raw::c_void,
    pub l_max: ::std::os::raw::c_int,
    pub num_sph_elements: ::std::os::raw::c_int,
    pub sph_coeffs: *const ::std::os::raw::c_void,
    pub coeff_set_offsets: *const ::std::os::raw::c_longlong,
    pub coeff_set_l_max: *const ::std::os::raw::c_int,
    pub freq_to_coeff_set: *const ::std::os::raw::c_int,
}
extern "C" {
    pub fn gpu_ska_calc_jones(
        model: *const SkaGpuModel,
        d_azs: *const f64,
        d_zas: *const f64,
        num_directions: ::std::os::raw::c_int,
        d_freqs_hz: *const f64,
        num_freqs: ::std::os::raw::c_int,
        num_tiles: ::std::os::raw::c_int,
        d_element_ids: *const ::std::os::raw::c_int,
        num_station_elements: ::std::os::raw::c_int,
        d_positions: *const f64,
        d_weights: *const f64,
        d_weights_y: *const f64,
        d_rotations: *const f64,
        d_frames: *const f64,
        pointing_az: f64,
        pointing_za: f64,
        d_latitude_rad: *const f64,
        iau_order: ::std::os::raw::c_int,
        d_results: *mut ::std::os::raw::c_void,
    ) -> *const ::std::os::raw::c_char;
}
