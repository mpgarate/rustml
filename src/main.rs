// reference
// https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/

fn predict(row: &Vec<f64>, coefficients: &Vec<f64>) -> f64 {
    let mut c_iter = coefficients.iter();

    let initial: f64 = *c_iter.next().unwrap();

    row.iter()
        .zip(c_iter)
        .fold(initial, |acc, (v, c)| acc + v * c)
}

fn coefficients_sgd(data: &Vec<Vec<f64>>, learn_rate: f64, iterations: usize) -> Vec<f64> {
    let mut coefficients = vec![0f64; data.get(0).unwrap().len()];

    for i in 0..iterations {
        let mut sum_error = 0f64;

        for row in data.iter() {
            let expected = row.iter().last().unwrap();
            let p = predict(&row, &coefficients);

            let error = p - expected;

            sum_error += error.powi(2);

            coefficients[0] = coefficients[0] - learn_rate * error;

            for j in 0..row.len() - 1 {
                coefficients[j + 1] = coefficients[j + 1] - learn_rate * error * row[j];
            }
        }

        println!("i: {:.2}, sum_error: {:.2}", i, sum_error);
    }

    return coefficients;
}

fn normalize(data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    use std::f64;

    let mut mins = vec![f64::INFINITY; data[0].len()];
    let mut maxes = vec![f64::NEG_INFINITY; data[0].len()];

    for line in data.iter() {
        for (i, &val) in line.iter().enumerate() {
            if val > maxes[i] {
                maxes[i] = val
            }

            if val < mins[i] {
                mins[i] = val
            }
        }
    }

    data.iter()
        .map(|row| {
            row.iter()
                .zip(mins.iter().zip(maxes.iter()))
                .map(|(val, (min, max))| (val - min) / (max - min))
                .collect()
        })
        .collect()
}

fn main() {
    use std::io::{self, BufRead};

    let stdin = io::stdin();

    let mut dataset: Vec<Vec<f64>> = stdin
        .lock()
        .lines()
        .map(|s| {
            s.unwrap()
                .split(",")
                .map(|s| s.parse::<f64>().unwrap())
                .collect()
        })
        .collect();

    dataset = normalize(dataset);

    let coefficients = coefficients_sgd(&dataset, 0.01f64, 50);

    for row in dataset {
        let expected = row.iter().last().unwrap();
        let p = predict(&row, &coefficients);

        println!("predict: {}, expected: {}", p, expected);
    }
}
