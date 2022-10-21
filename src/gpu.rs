// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Convenience code for interfacing with GPUs.

use std::{
    ffi::{c_void, CStr},
    panic::Location,
};

use thiserror::Error;

#[cfg(feature = "cuda")]
use cuda_runtime_sys::{
    cudaDeviceSynchronize as gpuDeviceSynchronize, cudaError::cudaSuccess as gpuSuccess,
    cudaFree as gpuFree, cudaGetErrorString as gpuGetErrorString,
    cudaGetLastError as gpuGetLastError, cudaMalloc as gpuMalloc, cudaMemcpy as gpuMemcpy,
    cudaMemcpyKind::cudaMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    cudaMemcpyKind::cudaMemcpyHostToDevice as gpuMemcpyHostToDevice,
};
#[cfg(feature = "hip")]
use hip_sys::hiprt::{
    hipDeviceSynchronize as gpuDeviceSynchronize, hipError_t::hipSuccess as gpuSuccess,
    hipFree as gpuFree, hipGetErrorString as gpuGetErrorString, hipGetLastError as gpuGetLastError,
    hipMalloc as gpuMalloc, hipMemcpy as gpuMemcpy,
    hipMemcpyKind::hipMemcpyDeviceToHost as gpuMemcpyDeviceToHost,
    hipMemcpyKind::hipMemcpyHostToDevice as gpuMemcpyHostToDevice,
};

// Set a compile-time variable type (this makes things a lot cleaner than having
// a #[cfg(...)] on many struct members).
cfg_if::cfg_if! {
    if #[cfg(feature = "gpu-single")] {
        pub type GpuFloat = f32;
        pub type GpuComplex = num_complex::Complex32;
    } else {
        pub type GpuFloat = f64;
        pub type GpuComplex = num_complex::Complex64;
    }
}

/// A Rust-managed pointer to GPU device memory. When this is dropped,
/// [`gpuFree`] is called on the pointer.
#[derive(Debug)]
pub struct DevicePointer<T> {
    ptr: *mut T,

    /// The number of elements of `T` allocated against the pointer.
    ///
    /// Yeah, I know a pointer isn't an array.
    num_elements: usize,
}

impl<T> DevicePointer<T> {
    /// Allocate a number of bytes on the device.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    #[track_caller]
    pub unsafe fn malloc(size: usize) -> Result<DevicePointer<T>, GpuError> {
        let mut d_ptr = std::ptr::null_mut();
        gpuMalloc(&mut d_ptr, size);
        check_for_errors(GpuCall::Malloc)?;
        Ok(Self {
            ptr: d_ptr.cast(),
            num_elements: size / std::mem::size_of::<T>(),
        })
    }

    /// Get the number of elements of `T` that have been allocated on the
    /// device. The number of bytes allocated is `num_elements *
    /// std::mem::size_of::<T>()`.
    pub fn get_num_elements(&self) -> usize {
        self.num_elements
    }

    /// Copy a slice of data to the device. Any type is allowed, and the
    /// returned pointer is to the device memory.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    #[track_caller]
    pub unsafe fn copy_to_device(v: &[T]) -> Result<DevicePointer<T>, GpuError> {
        let size = std::mem::size_of_val(v);
        let d_ptr = Self::malloc(size)?;
        gpuMemcpy(
            d_ptr.get_mut() as *mut c_void,
            v.as_ptr().cast(),
            size,
            gpuMemcpyHostToDevice,
        );
        check_for_errors(GpuCall::CopyToDevice)?;
        Ok(d_ptr)
    }

    /// Copy a slice of data from the device. The amount of data copied depends
    /// on the length of `v`, so if in doubt, the size of the device allocation
    /// should be checked with `DevicePointer::get_num_elements`.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    #[track_caller]
    pub unsafe fn copy_from_device(&self, v: &mut [T]) -> Result<(), GpuError> {
        let size = std::mem::size_of_val(v);
        gpuMemcpy(
            v.as_mut_ptr().cast(),
            self.ptr.cast(),
            size,
            gpuMemcpyDeviceToHost,
        );
        check_for_errors(GpuCall::CopyFromDevice)
    }

    /// Overwrite the device memory allocated against this [`DevicePointer`]
    /// with new memory. The amount of memory `v` must match what is allocated
    /// on against this [`DevicePointer`].
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors attempt
    /// to catch problems but there are no guarantees.
    #[track_caller]
    pub unsafe fn overwrite(&mut self, v: &[T]) -> Result<(), GpuError> {
        if v.len() != self.num_elements {
            return Err(GpuError::SizeMismatch);
        }
        let size = std::mem::size_of_val(v);
        gpuMemcpy(
            self.get_mut() as *mut c_void,
            v.as_ptr().cast(),
            size,
            gpuMemcpyHostToDevice,
        );
        check_for_errors(GpuCall::CopyToDevice)
    }

    /// Get a const pointer to the device memory.
    pub fn get(&self) -> *const T {
        self.ptr as *const T
    }

    /// Get a mutable pointer to the device memory.
    pub fn get_mut(&self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for DevicePointer<T> {
    fn drop(&mut self) {
        unsafe {
            gpuFree(self.ptr.cast());
        }
    }
}

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("When overwriting, the new amount of memory did not equal the old amount")]
    SizeMismatch,

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: cudaMemcpy to device failed: {msg}")]
    CopyToDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: hipMemcpy to device failed: {msg}")]
    CopyToDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: cudaMemcpy from device failed: {msg}")]
    CopyFromDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: hipMemcpy from device failed: {msg}")]
    CopyFromDevice {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: cudaMalloc error: {msg}")]
    Malloc {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: hipMalloc error: {msg}")]
    Malloc {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "cuda")]
    #[error("{file}:{line}: CUDA kernel error: {msg}")]
    Kernel {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },

    #[cfg(feature = "hip")]
    #[error("{file}:{line}: HIP kernel error: {msg}")]
    Kernel {
        msg: Box<str>,
        file: &'static str,
        line: u32,
    },
}

#[derive(Clone, Copy)]
pub enum GpuCall {
    Malloc,
    CopyToDevice,
    CopyFromDevice,
}

/// Run [`gpuGetLastError`] and [`gpuDeviceSynchronize`]. If either of these
/// calls return an error, it is converted to a Rust error and returned from
/// this function. The single argument describes what the just-performed
/// operation was and makes the returned error a helpful one.
///
/// # Safety
///
/// This function interfaces directly with the CUDA/HIP API. Rust errors attempt
/// to catch problems but there are no guarantees.
#[track_caller]
unsafe fn check_for_errors(gpu_call: GpuCall) -> Result<(), GpuError> {
    // Only do a device sync if we're in debug mode, for performance.
    let debug_mode = matches!(std::env::var("DEBUG").as_deref(), Ok("true"));
    if debug_mode {
        let code = gpuDeviceSynchronize();
        if code != gpuSuccess {
            let c_str = CStr::from_ptr(gpuGetErrorString(code));
            let msg = c_str.to_str();
            #[cfg(feature = "cuda")]
            let msg = msg.unwrap_or("<cannot read CUDA error string>");
            #[cfg(feature = "hip")]
            let msg = msg.unwrap_or("<cannot read HIP error string>");
            let location = Location::caller();
            return Err(match gpu_call {
                GpuCall::Malloc => GpuError::Malloc {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },

                GpuCall::CopyToDevice => GpuError::CopyToDevice {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },

                GpuCall::CopyFromDevice => GpuError::CopyFromDevice {
                    msg: msg.into(),
                    file: location.file(),
                    line: location.line(),
                },
            });
        }
    }

    let code = gpuGetLastError();
    if code != gpuSuccess {
        let c_str = CStr::from_ptr(gpuGetErrorString(code));
        let msg = c_str.to_str();
        #[cfg(feature = "cuda")]
        let msg = msg.unwrap_or("<cannot read CUDA error string>");
        #[cfg(feature = "hip")]
        let msg = msg.unwrap_or("<cannot read HIP error string>");
        let location = Location::caller();
        return Err(match gpu_call {
            GpuCall::Malloc => GpuError::Malloc {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
            GpuCall::CopyToDevice => GpuError::CopyToDevice {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
            GpuCall::CopyFromDevice => GpuError::CopyFromDevice {
                msg: msg.into(),
                file: location.file(),
                line: location.line(),
            },
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;
    use serial_test::serial;

    #[test]
    fn copy_to_and_from_device_succeeds() {
        unsafe {
            let heap = vec![0_u32; 100];
            let result = DevicePointer::copy_to_device(&heap);
            assert!(result.is_ok(), "Couldn't copy data to device memory");
            let d_ptr = result.unwrap();
            let mut heap2 = vec![1_u32; d_ptr.get_num_elements()];
            let result = d_ptr.copy_from_device(&mut heap2);
            assert!(result.is_ok(), "Couldn't copy data from device memory");
            result.unwrap();
            assert_abs_diff_eq!(Array1::from(heap), Array1::from(heap2));

            const LEN: usize = 100;
            let stack = [0_u32; LEN];
            let result = DevicePointer::copy_to_device(&stack);
            assert!(result.is_ok(), "Couldn't copy data to device memory");
            let d_ptr = result.unwrap();
            let mut stack2 = [1_u32; LEN];
            let result = d_ptr.copy_from_device(&mut stack2);
            assert!(result.is_ok(), "Couldn't copy data from device memory");
            result.unwrap();
            assert_abs_diff_eq!(Array1::from(stack.to_vec()), Array1::from(stack2.to_vec()));

            // Verify the copy_from_behaviour behaviour that the number of
            // elements copied depends on the host's array length.
            let mut stack3 = [1_u32; 100];
            let result = d_ptr.copy_from_device(&mut stack3[..50]);
            assert!(result.is_ok(), "Couldn't copy data from device memory");
            result.unwrap();
            assert_abs_diff_eq!(Array1::from(stack3[..50].to_vec()), Array1::zeros(50));
            assert_abs_diff_eq!(Array1::from(stack3[50..].to_vec()), Array1::ones(50));
        }
    }

    #[test]
    #[serial]
    fn gpu_malloc_huge_fails() {
        let size = 1024_usize.pow(4); // 1 TB;
        let result: Result<DevicePointer<u8>, GpuError> = unsafe { DevicePointer::malloc(size) };
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        #[cfg(feature = "cuda")]
        assert!(err.ends_with("cudaMalloc error: out of memory"), "{err}");
        #[cfg(feature = "hip")]
        assert!(
            err.contains("hipMalloc error"),
            "Error string wasn't expected; got: {err}"
        );
    }

    #[test]
    fn copy_from_non_existent_pointer_fails() {
        let d_ptr: DevicePointer<u8> = DevicePointer {
            ptr: std::ptr::null_mut::<u8>(),
            num_elements: 1,
        };
        let mut dest = [0; 100];
        let result = unsafe { d_ptr.copy_from_device(&mut dest) };
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        #[cfg(feature = "cuda")]
        assert!(
            err.ends_with("cudaMemcpy from device failed: invalid argument"),
            "Error string wasn't expected; got: {err}"
        );
        #[cfg(feature = "hip")]
        assert!(
            err.contains("hipMemcpy from device failed"),
            "Error string wasn't expected; got: {err}"
        );
    }
}
