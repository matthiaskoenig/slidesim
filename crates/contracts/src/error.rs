use std::path::PathBuf;

/// Errors from loading and validating a contract.
#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    #[error("{path}: {source}", path = path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {message}", path = path.display())]
    Parse { path: PathBuf, message: String },
    #[error("unsupported contract version '{found}', expected '{expected}'")]
    UnsupportedVersion {
        found: String,
        expected: &'static str,
    },
    #[error("unsupported length unit '{found}', expected 'um'")]
    UnsupportedLengthUnit { found: String },
    /// A table or direction names a map the scene does not define.
    #[error("unknown map '{map}', the scene defines: [{}]", available.join(", "))]
    UnknownMap { map: String, available: Vec<String> },
    /// A value violates a contract rule; the message names the field.
    #[error("{0}")]
    Invalid(String),
}
