use std::f64::consts::PI;
use ndarray::*;
use ndarray_linalg::*;

const DT: f64 = 0.005;
const X_MIN: f64 = 0.0;
const X_MAX: f64 = 2.0;
const N: usize = 200;
const DX: f64 = (X_MAX - X_MIN) / N as f64;
const U: f64 = 1.0;
const M: usize = 200;

fn arr_to_save(save: &mut String, arr: &Array1<f64>) {
    for e in arr.iter() {
        *save = save.to_owned() + &(e.to_string()) + ",";
    }
    *save = save.to_owned() + "\n";
}

fn main() {
    let mut f: Array1<f64> = Array::from((0..N+1).map(|i| {
                                                     let x = i as f64 * DX;
                                                     if x <= 1.0 { (PI * x).sin() }
                                                     else { 0.0 }
                                                 })
                                                 .collect::<Vec<f64>>());
    let u: Array1<f64> = Array::from(vec![U; N+1]);
    let mut save = String::new();
    arr_to_save(&mut save, &f);
    
    let mut a: Array2<f64> = Array::zeros((N+1, N+1));
    let mut b: Array1<f64> = Array::zeros(N+1);
    for _ in 0..M {
        a.fill(0.0);
        b.fill(0.0);
        for i in 0..N {
            let f1 = f[i];
            let f2 = f[i + 1];
            let u1 = u[i];
            let u2 = u[i + 1];
            let tau = 1.0 / ((2.0/DT).powi(2) + (2.0 * U / DX).powi(2)).sqrt();

            let m_core = 1.0 / DT / 6.0 * DX;
            a[[i    , i    ]] += 2.0 * m_core;
            a[[i    , i + 1]] += 1.0 * m_core;
            a[[i + 1, i    ]] += 1.0 * m_core;
            a[[i + 1, i + 1]] += 2.0 * m_core;

            let md_core = tau / DT / 6.0;
            a[[i    , i    ]] += md_core * (-2.0 * u1 -       u2);
            a[[i    , i + 1]] += md_core * (-      u1 - 2.0 * u2);
            a[[i + 1, i    ]] += md_core * ( 2.0 * u1 +       u2);
            a[[i + 1, i + 1]] += md_core * (       u1 + 2.0 * u2);

            let s_core = 0.5 / 6.0;
            a[[i    , i    ]] += s_core * (-2.0 * u1 -       u2);
            a[[i    , i + 1]] += s_core * ( 2.0 * u1 +       u2);
            a[[i + 1, i    ]] += s_core * (-      u1 - 2.0 * u2);
            a[[i + 1, i + 1]] += s_core * (       u1 + 2.0 * u2);

            let sd_core = 0.5 * tau / 3.0 / DX * (u1.powi(2) + u1 * u2 + u2.powi(2));
            a[[i    , i    ]] += sd_core;
            a[[i    , i + 1]] -= sd_core;
            a[[i + 1, i    ]] -= sd_core;
            a[[i + 1, i + 1]] += sd_core;

            let m_core = 1.0 / DT * DX / 6.0;
            b[i    ] += m_core * (2.0 * f1 +       f2);
            b[i + 1] += m_core * (      f1 + 2.0 * f2);

            let md_core = tau / DT / 6.0;
            b[i    ] += md_core * ((-2.0 * u1 - u2) * f1 + (-u1 - 2.0 * u2) * f2);
            b[i + 1] += md_core * (( 2.0 * u1 + u2) * f1 + ( u1 + 2.0 * u2) * f2);

            let s_core = 0.5 / 6.0;
            b[i    ] -= s_core * ((-2.0 * u1 -       u2) * f1 + (2.0 * u1 +       u2) * f2);
            b[i + 1] -= s_core * ((-      u1 - 2.0 * u2) * f1 + (      u1 + 2.0 * u2) * f2);

            let sd_core = 0.5 * tau / DX / 3.0 * (u1.powi(2) + u1 * u2 + u2.powi(2));
            b[i    ] -= sd_core * ( f1 - f2);
            b[i + 1] -= sd_core * (-f1 + f2);

            a[[0, i]] = 0.0;
            a[[N, i]] = 0.0;
        }
        a[[0, 0]] = 1.0;
        a[[N, N]] = 1.0;
        b[0] = 0.0;
        b[N] = 0.0;

        f = a.solve(&b).unwrap();
        arr_to_save(&mut save, &f);
    }

    println!("{}", save);
}