use pyo3::prelude::*;

mod neighbors;
mod scalers;
mod utils;

use neighbors::KNeighborsRegressor;
use scalers::{MinMaxScaler, StandardScaler};

#[pymodule]
fn _rklearn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StandardScaler>()?;
    m.add_class::<MinMaxScaler>()?;
    m.add_class::<KNeighborsRegressor>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
