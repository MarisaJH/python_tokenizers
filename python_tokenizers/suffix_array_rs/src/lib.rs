use pyo3::prelude::*;
use esaxx_rs;

#[pyfunction]
fn suffix(text: &str) -> PyResult<Vec<(String, i32)>> {
    let suffix_result = esaxx_rs::suffix(text).unwrap();
    
    // Convert Suffix to Vec<(String, i32)>
    let result: Vec<(String, i32)> = suffix_result.iter()
        .map(|(chars, freq)| {
            let substring: String = chars.iter().collect();
            (substring, freq as i32)
        })
        .collect();

    Ok(result)
}

#[pymodule]
fn suffix_array_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(suffix, m)?)?;
    Ok(())
}