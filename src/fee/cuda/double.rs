/* automatically generated by rust-bindgen 0.68.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct FEECoeffs {
    pub x_q1_accum: *const f64,
    pub x_q2_accum: *const f64,
    pub x_m_accum: *const i8,
    pub x_n_accum: *const i8,
    pub x_m_signs: *const i8,
    pub x_m_abs_m: *const i8,
    pub x_lengths: *const ::std::os::raw::c_int,
    pub x_offsets: *const ::std::os::raw::c_int,
    pub y_q1_accum: *const f64,
    pub y_q2_accum: *const f64,
    pub y_m_accum: *const i8,
    pub y_n_accum: *const i8,
    pub y_m_signs: *const i8,
    pub y_m_abs_m: *const i8,
    pub y_lengths: *const ::std::os::raw::c_int,
    pub y_offsets: *const ::std::os::raw::c_int,
    pub n_max: ::std::os::raw::c_uchar,
}
#[test]
fn bindgen_test_layout_FEECoeffs() {
    const UNINIT: ::std::mem::MaybeUninit<FEECoeffs> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<FEECoeffs>(),
        136usize,
        concat!("Size of: ", stringify!(FEECoeffs))
    );
    assert_eq!(
        ::std::mem::align_of::<FEECoeffs>(),
        8usize,
        concat!("Alignment of ", stringify!(FEECoeffs))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_q1_accum) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_q1_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_q2_accum) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_q2_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_m_accum) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_m_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_n_accum) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_n_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_m_signs) as usize - ptr as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_m_signs)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_m_abs_m) as usize - ptr as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_m_abs_m)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_lengths) as usize - ptr as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_lengths)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).x_offsets) as usize - ptr as usize },
        56usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(x_offsets)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_q1_accum) as usize - ptr as usize },
        64usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_q1_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_q2_accum) as usize - ptr as usize },
        72usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_q2_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_m_accum) as usize - ptr as usize },
        80usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_m_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_n_accum) as usize - ptr as usize },
        88usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_n_accum)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_m_signs) as usize - ptr as usize },
        96usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_m_signs)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_m_abs_m) as usize - ptr as usize },
        104usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_m_abs_m)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_lengths) as usize - ptr as usize },
        112usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_lengths)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).y_offsets) as usize - ptr as usize },
        120usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(y_offsets)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).n_max) as usize - ptr as usize },
        128usize,
        concat!(
            "Offset of field: ",
            stringify!(FEECoeffs),
            "::",
            stringify!(n_max)
        )
    );
}
extern "C" {
    pub fn cuda_calc_jones(
        d_azs: *const f64,
        d_zas: *const f64,
        num_directions: ::std::os::raw::c_int,
        d_coeffs: *const FEECoeffs,
        num_coeffs: ::std::os::raw::c_int,
        d_norm_jones: *const ::std::os::raw::c_void,
        d_array_latitude_rad: *const f64,
        iau_reorder: ::std::os::raw::c_int,
        d_results: *mut ::std::os::raw::c_void,
    ) -> *const ::std::os::raw::c_char;
}
