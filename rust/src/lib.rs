use pyo3::prelude::*;

mod scalers;
mod utils;

use scalers::{MinMaxScaler, StandardScaler};

#[pymodule]
fn _rklearn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StandardScaler>()?;
    m.add_class::<MinMaxScaler>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
