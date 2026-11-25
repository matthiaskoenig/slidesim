//! The global options database: single options, and options scoped to a prefix and a lifetime.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::ptr::{null, null_mut};

use crate::{Error, Result, alive, c_string, chk, sys};

/// The value of the option `name` in the global options database (see [`crate::Petsc::option`]).
pub(crate) fn get(name: &str) -> Result<Option<String>> {
    let name = c_string(name, "PETSc option")?;
    let mut value: *const c_char = null();
    let mut set = false;
    // SAFETY: `value` points into the options database, and is copied before any other call.
    chk!(unsafe {
        sys::PetscOptionsFindPair(null_mut(), null(), name.as_ptr(), &mut value, &mut set)
    });
    Ok(set.then(|| lossy(value)))
}

/// Sets the option `name` in the global options database (see [`crate::Petsc::set_option`]).
pub(crate) fn set(name: &str, value: Option<&str>) -> Result<()> {
    if !name.starts_with('-') {
        return Err(Error::new(
            sys::PETSC_ERR_ARG_WRONG,
            format!("PETSc option {name:?} does not start with '-'"),
        ));
    }
    let name = c_string(name, "PETSc option")?;
    let value = value
        .map(|v| c_string(v, "PETSc option value"))
        .transpose()?;
    // SAFETY: PETSc copies both strings.
    chk!(unsafe {
        sys::PetscOptionsSetValue(
            null_mut(),
            name.as_ptr(),
            value.as_ref().map_or(null(), |v| v.as_ptr()),
        )
    });
    Ok(())
}

/// Removes the option `name` from the global options database.
pub(crate) fn clear(name: &str) -> Result<()> {
    let name = c_string(name, "PETSc option")?;
    chk!(unsafe { sys::PetscOptionsClearValue(null_mut(), name.as_ptr()) });
    Ok(())
}

/// Sets the option `name` back to `previous`, as [`get`] returned it.
pub(crate) fn restore(name: &str, previous: Option<&str>) -> Result<()> {
    match previous {
        None => clear(name),
        Some("") => set(name, None),
        Some(v) => set(name, Some(v)),
    }
}

/// A C string of PETSc as a `String` (empty for null).
fn lossy(s: *const c_char) -> String {
    if s.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}

thread_local! {
    /// The prefixes of the live [`OptionsScope`]s, in lower case (option names are case
    /// insensitive).
    static LIVE: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
}

/// Options in the global options database under a prefix, for as long as the scope lives: the
/// options of the PETSc objects with that prefix (`TSSetOptionsPrefix` and the like), and of no
/// other object. Dropping the scope sets each of its options back to what it was before (unset,
/// as a rule).
///
/// One scope per prefix lives at a time, so that objects of different scopes never read each
/// other's options.
#[derive(Debug)]
pub struct OptionsScope {
    /// The key of the prefix in [`LIVE`].
    key: String,
    /// The full names of the options set (`-prefix_name`) and their previous values, as [`get`]
    /// returned them.
    previous: Vec<(String, Option<String>)>,
    _not_send: PhantomData<*const ()>,
}

impl OptionsScope {
    /// Sets the options of `texts` under `prefix`: each text is an options string as on the
    /// command line (for example `-ksp_type cg -ksp_monitor`, read by `PetscOptionsInsertString`,
    /// so `-options_file` works too), and a later text overrides the options of an earlier one.
    /// `-ksp_type cg` under the prefix `u_` sets `-u_ksp_type cg`.
    ///
    /// Errors: `prefix` does not start with a letter or has characters other than ASCII letters,
    /// digits and `_`, (`PETSC_ERR_ARG_WRONG`); a scope of `prefix` is alive
    /// (`PETSC_ERR_ARG_WRONGSTATE`); PETSc cannot read a text.
    pub fn new(prefix: &str, texts: &[&str]) -> Result<OptionsScope> {
        let valid = prefix.starts_with(|c: char| c.is_ascii_alphabetic())
            && prefix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !valid {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_WRONG,
                format!(
                    "options prefix {prefix:?}: it must start with a letter and have ASCII \
                     letters, digits and '_' only"
                ),
            ));
        }
        let key = prefix.to_ascii_lowercase();
        if LIVE.with(|live| live.borrow().contains(&key)) {
            return Err(Error::new(
                sys::PETSC_ERR_ARG_WRONGSTATE,
                format!("options prefix {prefix:?} is in use by another scope"),
            ));
        }
        let options = parse(prefix, texts)?;
        LIVE.with(|live| live.borrow_mut().insert(key.clone()));
        let mut scope = OptionsScope {
            key,
            previous: Vec::with_capacity(options.len()),
            _not_send: PhantomData,
        };
        // On an error, dropping the scope restores the options set so far.
        for (name, value) in options {
            let name = format!("-{name}");
            let previous = get(&name)?;
            set(&name, value.as_deref())?;
            scope.previous.push((name, previous));
        }
        Ok(scope)
    }
}

impl Drop for OptionsScope {
    fn drop(&mut self) {
        if alive() {
            for (name, previous) in self.previous.iter().rev() {
                // Nothing to report an error to; the other options are still restored.
                let _ = restore(name, previous.as_deref());
            }
        }
        LIVE.with(|live| live.borrow_mut().remove(&self.key));
    }
}

/// The options of `texts` under `prefix` (names without the leading `-`, and values), as PETSc
/// reads them: into a database of their own, all of whose options are unused
/// (`PetscOptionsLeftGet`).
fn parse(prefix: &str, texts: &[&str]) -> Result<Vec<(String, Option<String>)>> {
    let prefix = c_string(prefix, "options prefix")?;
    let texts = texts
        .iter()
        .map(|t| c_string(t, "PETSc options"))
        .collect::<Result<Vec<_>>>()?;
    let mut db: sys::PetscOptions = null_mut();
    chk!(unsafe { sys::PetscOptionsCreate(&mut db) });
    let parsed = (|| {
        chk!(unsafe { sys::PetscOptionsPrefixPush(db, prefix.as_ptr()) });
        for text in &texts {
            chk!(unsafe { sys::PetscOptionsInsertString(db, text.as_ptr()) });
        }
        chk!(unsafe { sys::PetscOptionsPrefixPop(db) });
        let mut n = 0;
        let (mut names, mut values) = (null_mut(), null_mut());
        // SAFETY: `n` names and values (null for an option without a value), borrowed from
        // `db` until the restore.
        chk!(unsafe { sys::PetscOptionsLeftGet(db, &mut n, &mut names, &mut values) });
        let options = (0..n as usize)
            .map(|i| {
                let (name, value) = unsafe { (*names.add(i), *values.add(i)) };
                (lossy(name), (!value.is_null()).then(|| lossy(value)))
            })
            .collect();
        chk!(unsafe { sys::PetscOptionsLeftRestore(db, &mut n, &mut names, &mut values) });
        Ok(options)
    })();
    // Cannot fail: the prefix stack does not matter, and nothing was pushed.
    unsafe { sys::PetscOptionsDestroy(&mut db) };
    parsed
}
