use pyo3::prelude::*;

mod linear;
mod neighbors;
mod scalers;
mod utils;

use linear::RidgeRegressor;
use neighbors::KNeighborsRegressor;
use scalers::{MinMaxScaler, StandardScaler};

#[pymodule]
fn _rklearn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StandardScaler>()?;
    m.add_class::<MinMaxScaler>()?;
    m.add_class::<KNeighborsRegressor>()?;
    m.add_class::<RidgeRegressor>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
