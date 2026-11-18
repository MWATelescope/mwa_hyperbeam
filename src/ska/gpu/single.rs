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
    pub phis: *const f32,
    pub num_phis: ::std::os::raw::c_int,
    pub thetas: *const f32,
    pub num_thetas: ::std::os::raw::c_int,
    pub planes: *const ::std::os::raw::c_void,
    pub l_max: ::std::os::raw::c_int,
    pub num_coeffs: ::std::os::raw::c_int,
    pub num_sph_elements: ::std::os::raw::c_int,
    pub sph_coeffs: *const ::std::os::raw::c_void,
    pub freq_to_coeff_set: *const ::std::os::raw::c_int,
}
extern "C" {
    pub fn gpu_ska_calc_jones(
        model: *const SkaGpuModel,
        d_azs: *const f32,
        d_zas: *const f32,
        num_directions: ::std::os::raw::c_int,
        d_freqs_hz: *const f32,
        num_freqs: ::std::os::raw::c_int,
        num_tiles: ::std::os::raw::c_int,
        d_element_ids: *const ::std::os::raw::c_int,
        num_station_elements: ::std::os::raw::c_int,
        d_positions: *const f32,
        d_weights: *const f32,
        d_rotations: *const f32,
        pointing_az: f32,
        pointing_za: f32,
        d_latitude_rad: *const f32,
        iau_order: ::std::os::raw::c_int,
        d_results: *mut ::std::os::raw::c_void,
    ) -> *const ::std::os::raw::c_char;
}
