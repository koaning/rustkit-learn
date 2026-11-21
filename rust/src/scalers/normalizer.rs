#![allow(deprecated)]

use ndarray::{Array2, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone, Copy)]
enum Norm {
    L1,
    L2,
    Max,
}

impl Norm {
    fn parse(s: &str) -> PyResult<Self> {
        match s {
            "l1" => Ok(Norm::L1),
            "l2" => Ok(Norm::L2),
            "max" => Ok(Norm::Max),
            _ => Err(PyErr::new::<PyValueError, _>(format!(
                "Unsupported norm '{}': expected 'l1', 'l2' or 'max'",
                s
            ))),
        }
    }
}

fn normalize_rows(data: ArrayView2<f64>, norm: Norm) -> Array2<f64> {
    let mut out = data.to_owned();
    Zip::from(out.rows_mut()).par_for_each(|mut row| {
        let n = match norm {
            Norm::L1 => row.iter().map(|v| v.abs()).sum::<f64>(),
            Norm::L2 => row.iter().map(|v| v * v).sum::<f64>().sqrt(),
            Norm::Max => row.iter().fold(0.0_f64, |m, v| m.max(v.abs())),
        };
        // Rows with zero norm are left unchanged, like sklearn.
        if n != 0.0 {
            row.mapv_inplace(|v| v / n);
        }
    });
    out
}

/// Normalize samples individually to unit norm.
///
/// Each sample (i.e. each row of the data matrix) with at least one non zero
/// component is rescaled independently of other samples so that its norm
/// (l1, l2 or max) equals one. This transformer is stateless.
///
/// Parameters
/// ----------
/// norm : {'l1', 'l2', 'max'}, default='l2'
///     The norm to use to normalize each non zero sample.
#[pyclass]
pub struct Normalizer {
    norm: String,
    n_features_in: Option<usize>,
}

#[pymethods]
impl Normalizer {
    #[new]
    #[pyo3(signature = (norm="l2"))]
    fn new(norm: &str) -> PyResult<Self> {
        Norm::parse(norm)?;
        Ok(Normalizer {
            norm: norm.to_string(),
            n_features_in: None,
        })
    }

    /// Only validates the input; Normalizer is stateless.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyRefMut<'py, Self> {
        slf.n_features_in = Some(x.as_array().ncols());
        slf
    }

    /// Scale each non zero row of X to unit norm.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let norm = Norm::parse(&self.norm)?;
        let arr = x.as_array().to_owned();
        if let Some(n) = self.n_features_in {
            if n != arr.ncols() {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "X has {} features, but Normalizer is expecting {} features",
                    arr.ncols(),
                    n
                )));
            }
        }
        let out = py.allow_threads(move || normalize_rows(arr.view(), norm));
        Ok(out.into_pyarray(py))
    }

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fitted = Self::fit(slf, x.clone());
        fitted.transform(py, x)
    }

    /// Number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Get parameters for this estimator.
    #[pyo3(signature = (deep=true))]
    fn get_params<'py>(&self, py: Python<'py>, deep: bool) -> PyResult<Bound<'py, PyDict>> {
        let _ = deep;
        let d = PyDict::new(py);
        d.set_item("norm", &self.norm)?;
        Ok(d)
    }

    /// Set the parameters of this estimator.
    #[pyo3(signature = (**params))]
    fn set_params<'py>(
        mut slf: PyRefMut<'py, Self>,
        params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Some(p) = params {
            for (k, v) in p.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "norm" => {
                        let norm: String = v.extract()?;
                        Norm::parse(&norm)?;
                        slf.norm = norm;
                    }
                    _ => {
                        return Err(PyErr::new::<PyValueError, _>(format!(
                            "Invalid parameter {} for estimator Normalizer",
                            key
                        )))
                    }
                }
            }
        }
        Ok(slf)
    }
}
