// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! General FFI code for error handling.
//!
//! Most of this is derived from
//! <https://michael-f-bryan.github.io/rust-ffi-guide/errors/return_types.html

use std::{
    cell::RefCell,
    os::raw::{c_char, c_int},
    slice,
};

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = RefCell::new(None);
}

/// Update the most recent error, clearing whatever may have been there before.
pub(crate) fn update_last_error(err: String) {
    LAST_ERROR.with(|prev| {
        *prev.borrow_mut() = Some(err);
    });
}

/// Retrieve the most recent error, clearing it in the process.
fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|prev| prev.borrow_mut().take())
}

/// Calculate the number of bytes in the last error's error message **not**
/// including any trailing `null` characters.
#[no_mangle]
pub extern "C" fn hb_last_error_length() -> c_int {
    LAST_ERROR.with(|prev| match *prev.borrow() {
        Some(ref err) => err.to_string().len() as c_int + 1,
        None => 0,
    })
}

macro_rules! ffi_error {
    ($result:expr) => {{
        match $result {
            Ok(r) => r,
            Err(e) => {
                update_last_error(e.to_string());
                return 1;
            }
        }
    }};
}
pub(crate) use ffi_error;

/// Write the most recent error message into a caller-provided buffer as a UTF-8
/// string, returning the number of bytes written.
///
/// # Note
///
/// This writes a **UTF-8** string into the buffer. Windows users may need to
/// convert it to a UTF-16 "unicode" afterwards.
///
/// If there are no recent errors then this returns `0` (because we wrote 0
/// bytes). `-1` is returned if there are any errors, for example when passed a
/// null pointer or a buffer of insufficient size.
#[no_mangle]
pub unsafe extern "C" fn hb_last_error_message(buffer: *mut c_char, length: c_int) -> c_int {
    if buffer.is_null() {
        // warn!("Null pointer passed into last_error_message() as the buffer");
        return -1;
    }

    let last_error = match take_last_error() {
        Some(err) => err,
        None => return 0,
    };

    let buffer = slice::from_raw_parts_mut(buffer as *mut u8, length as usize);

    if last_error.len() >= buffer.len() {
        // warn!("Buffer provided for writing the last error message is too small.");
        // warn!(
        //     "Expected at least {} bytes but got {}",
        //     last_error.len() + 1,
        //     buffer.len()
        // );
        return -1;
    }

    std::ptr::copy_nonoverlapping(last_error.as_ptr(), buffer.as_mut_ptr(), last_error.len());

    // Add a trailing null so people using the string as a `char *` don't
    // accidentally read into garbage.
    buffer[last_error.len()] = 0;

    last_error.len() as c_int
}
