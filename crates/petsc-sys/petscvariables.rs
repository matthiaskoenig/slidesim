//! PETSc's make variables (`$PETSC_DIR/lib/petsc/conf/petscvariables`) and the compiler and
//! linker flags in them, for example `PTSCOTCH_INCLUDE` and `PTSCOTCH_LIB`: where an external
//! package PETSc was built with lives, which need not be inside `PETSC_DIR` (`--with-<pkg>-dir`,
//! Spack, conda-forge, cluster modules).
//!
//! Standard library only: build scripts include it (`#[path]`), and `tests/petscvariables.rs`
//! tests it.

use std::path::{Path, PathBuf};

/// The compile and link flags of an external package of a PETSc installation.
#[derive(Debug, PartialEq, Eq)]
pub struct Package {
    /// `<PACKAGE>_INCLUDE`; empty if the headers are in the compiler's default directories.
    pub include: Flags,
    /// `<PACKAGE>_LIB`; names at least one library.
    pub lib: Flags,
}

/// The files [`package`] reads in the PETSc installation `petsc_dir`, for
/// `cargo:rerun-if-changed`.
pub fn files(petsc_dir: &Path) -> [PathBuf; 2] {
    [
        petsc_dir.join("include/petscconf.h"),
        petsc_dir.join("lib/petsc/conf/petscvariables"),
    ]
}

/// The flags of the external package `name` (PETSc's name in capitals, for example `PTSCOTCH`) of
/// the PETSc installation `petsc_dir`: `None` if its `petscconf.h` does not define
/// `PETSC_HAVE_<name>`. An error (a message) if a file cannot be read, or if PETSc has the package
/// but `petscvariables` lacks `<name>_INCLUDE` or `<name>_LIB`, or the latter names no library.
pub fn package(petsc_dir: &Path, name: &str) -> Result<Option<Package>, String> {
    let [conf, variables] = files(petsc_dir);
    let read = |path: &Path| {
        std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
    };
    let define = format!("#define PETSC_HAVE_{name} 1");
    if !read(&conf)?.lines().any(|l| l.trim() == define) {
        return Ok(None);
    }
    let text = read(&variables)?;
    let flags = |suffix: &str| {
        let var = format!("{name}_{suffix}");
        variable(&text, &var)
            .map(|v| parse_flags(&v))
            .ok_or_else(|| {
                format!(
                    "PETSc has {name} ({}), but {} does not define {var}",
                    conf.display(),
                    variables.display()
                )
            })
    };
    let package = Package {
        include: flags("INCLUDE")?,
        lib: flags("LIB")?,
    };
    if package.lib.libs.is_empty() {
        return Err(format!(
            "PETSc has {name}, but {name}_LIB in {} names no library",
            variables.display()
        ));
    }
    Ok(Some(package))
}

/// The value of the make variable `name` in the text of a `petscvariables` file: the last line
/// `name = value` (or `:=`, `?=`), with continuation lines (ending in `\`) joined. `None` if
/// there is no such line.
pub fn variable(text: &str, name: &str) -> Option<String> {
    let mut value = None;
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let mut line = line.to_string();
        while line.ends_with('\\') {
            line.pop();
            line.push(' ');
            match lines.next() {
                Some(next) => line.push_str(next),
                None => break,
            }
        }
        let Some((key, rest)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim_end();
        let key = key
            .strip_suffix(':')
            .or_else(|| key.strip_suffix('?'))
            .unwrap_or(key)
            .trim();
        if key == name {
            value = Some(rest.trim().to_string());
        }
    }
    value
}

/// The flags of a compile or link line, sorted by kind, each in the order of the line.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// `-I<dir>` or `-I <dir>`.
    pub include_dirs: Vec<String>,
    /// `-L<dir>` or `-L <dir>`, and the directories of libraries given as files.
    pub link_dirs: Vec<String>,
    /// `-l<name>`, and the names of libraries given as files (`/dir/lib<name>.so` -> `<name>`).
    pub libs: Vec<String>,
    /// `-Wl,-rpath,<dir>`, `-Wl,-rpath=<dir>` or `-Wl,-R,<dir>`.
    pub rpaths: Vec<String>,
    /// Everything else, for example `-pthread` or `-Wl,--as-needed`: for the caller to warn about.
    pub ignored: Vec<String>,
}

/// The flags of the compile or link line `line` (whitespace separated, no quoting).
pub fn parse_flags(line: &str) -> Flags {
    let mut flags = Flags::default();
    let mut tokens = line.split_whitespace();
    while let Some(token) = tokens.next() {
        // `-I dir` and `-L dir`: the directory is the next token.
        let mut joined = |prefix: &str| -> Option<String> {
            let rest = token.strip_prefix(prefix)?;
            if rest.is_empty() {
                tokens.next().map(String::from)
            } else {
                Some(rest.to_string())
            }
        };
        if let Some(dir) = joined("-I") {
            flags.include_dirs.push(dir);
        } else if let Some(dir) = joined("-L") {
            flags.link_dirs.push(dir);
        } else if let Some(name) = token.strip_prefix("-l")
            && !name.is_empty()
            && !name.starts_with(':')
        {
            flags.libs.push(name.to_string());
        } else if let Some(args) = token.strip_prefix("-Wl,")
            && let Some(dirs) = rpaths(args)
        {
            flags.rpaths.extend(dirs);
        } else if let Some((dir, name)) = library_file(token) {
            flags.link_dirs.push(dir);
            flags.libs.push(name);
        } else {
            flags.ignored.push(token.to_string());
        }
    }
    flags
}

/// The directories of the linker arguments `args` (of `-Wl,<args>`) if they are rpath options
/// only: `-rpath,<dir>`, `-rpath=<dir>`, `-R,<dir>`, several of them. `None` for anything else.
fn rpaths(args: &str) -> Option<Vec<String>> {
    let mut dirs = Vec::new();
    let mut parts = args.split(',');
    while let Some(part) = parts.next() {
        let dir = if part == "-rpath" || part == "-R" || part == "--rpath" {
            parts.next()?
        } else {
            part.strip_prefix("-rpath=")
                .or_else(|| part.strip_prefix("--rpath="))?
        };
        if dir.is_empty() {
            return None;
        }
        dirs.push(dir.to_string());
    }
    (!dirs.is_empty()).then_some(dirs)
}

/// The directory and the name of a library given as an absolute path to its file,
/// `<dir>/lib<name>.so`, `.so.<version>`, `.a` or `.dylib`.
fn library_file(token: &str) -> Option<(String, String)> {
    let path = Path::new(token);
    if !path.is_absolute() {
        return None;
    }
    let file = path.file_name()?.to_str()?;
    let stem = file.strip_prefix("lib")?;
    let name = if let Some((name, _)) = stem.split_once(".so") {
        // `.so` or `.so.<version>`, not some other name containing `.so`.
        let rest = &stem[name.len() + 3..];
        if !(rest.is_empty() || rest.starts_with('.')) {
            return None;
        }
        name
    } else {
        stem.strip_suffix(".a")
            .or_else(|| stem.strip_suffix(".dylib"))?
    };
    if name.is_empty() {
        return None;
    }
    Some((path.parent()?.to_str()?.to_string(), name.to_string()))
}
