//! `CeedElemRestriction`: maps between L-vectors (global numbering) and E-vectors (per element).

use std::ptr::null_mut;

use crate::{Ceed, Error, Result, chk, sys, to_int, to_size};

/// A libCEED element restriction (`CeedElemRestriction`).
#[derive(Debug)]
pub struct ElemRestriction {
    raw: sys::CeedElemRestriction,
    ceed: Ceed,
    l_size: usize,
}

impl ElemRestriction {
    /// A restriction of `num_elem` elements with `elem_size` nodes and `num_comp` components:
    /// component `c` of node `i` of element `e` is entry `offsets[e * elem_size + i] + c *
    /// comp_stride` of an L-vector of length `l_size` (`CeedElemRestrictionCreate`, with a copy
    /// of the offsets).
    ///
    /// The offsets are checked here (the optimized backends do not check them, and an offset
    /// out of range would make them access memory outside the vector).
    pub fn new(
        ceed: &Ceed,
        num_elem: usize,
        elem_size: usize,
        num_comp: usize,
        comp_stride: usize,
        l_size: usize,
        offsets: &[i32],
    ) -> Result<ElemRestriction> {
        let dimension = |message: String| Error::new(sys::CEED_ERROR_DIMENSION, message);
        if num_comp == 0 || elem_size == 0 {
            return Err(dimension(format!(
                "element restriction with {num_comp} components and {elem_size} nodes per \
                 element (need at least 1 each)"
            )));
        }
        if num_elem.checked_mul(elem_size) != Some(offsets.len()) {
            return Err(dimension(format!(
                "{} offsets for {num_elem} elements of {elem_size} nodes",
                offsets.len()
            )));
        }
        let last_comp = (num_comp - 1)
            .checked_mul(comp_stride)
            .ok_or_else(|| dimension("component stride overflows".to_string()))?;
        if let Some((i, &o)) = offsets.iter().enumerate().find(|&(_, &o)| {
            usize::try_from(o).map_or(true, |o| o.saturating_add(last_comp) >= l_size)
        }) {
            return Err(dimension(format!(
                "restriction offset {i} ({o}, last component at {o} + {last_comp}) out of range \
                 [0, {l_size})"
            )));
        }
        let mut raw: sys::CeedElemRestriction = null_mut();
        // SAFETY: CEED_COPY_VALUES copies `num_elem * elem_size` offsets (checked above).
        chk!(ceed, unsafe {
            sys::CeedElemRestrictionCreate(
                ceed.as_raw(),
                to_int(num_elem, "number of elements")?,
                to_int(elem_size, "element size")?,
                to_int(num_comp, "number of components")?,
                to_int(comp_stride, "component stride")?,
                to_size(l_size, "L-vector size")?,
                sys::CEED_MEM_HOST,
                sys::CEED_COPY_VALUES,
                offsets.as_ptr(),
                &mut raw,
            )
        });
        Ok(ElemRestriction {
            raw,
            ceed: ceed.clone(),
            l_size,
        })
    }

    /// A strided restriction in the backend's layout (`CEED_STRIDES_BACKEND`): data stored per
    /// element, such as values at quadrature points, in an L-vector of length `l_size`
    /// (at least `num_elem * elem_size * num_comp`, which libCEED checks).
    pub fn strided(
        ceed: &Ceed,
        num_elem: usize,
        elem_size: usize,
        num_comp: usize,
        l_size: usize,
    ) -> Result<ElemRestriction> {
        let mut raw: sys::CeedElemRestriction = null_mut();
        // The address of CEED_STRIDES_BACKEND itself, which libCEED recognizes.
        let strides = (&raw const sys::CEED_STRIDES_BACKEND).cast::<sys::CeedInt>();
        chk!(ceed, unsafe {
            sys::CeedElemRestrictionCreateStrided(
                ceed.as_raw(),
                to_int(num_elem, "number of elements")?,
                to_int(elem_size, "element size")?,
                to_int(num_comp, "number of components")?,
                to_size(l_size, "L-vector size")?,
                strides,
                &mut raw,
            )
        });
        Ok(ElemRestriction {
            raw,
            ceed: ceed.clone(),
            l_size,
        })
    }

    /// The length of the L-vector.
    pub fn l_size(&self) -> usize {
        self.l_size
    }

    /// The raw handle, for the libCEED API not wrapped here.
    pub fn as_raw(&self) -> sys::CeedElemRestriction {
        self.raw
    }

    pub(crate) fn ceed(&self) -> &Ceed {
        &self.ceed
    }
}

impl Drop for ElemRestriction {
    fn drop(&mut self) {
        // Nothing to report an error to; operators using it hold their own reference.
        unsafe { sys::CeedElemRestrictionDestroy(&mut self.raw) };
    }
}
