use std::f64::INFINITY;
use ndarray::*;
use ndarray_linalg::*;
use gnuplot::*;

pub const X_MIN: f64 = -1.0;
pub const X_MAX: f64 =  1.0;

pub const NODE_TOTAL: usize = 4;
pub const ELE_TOTAL: usize = NODE_TOTAL - 1;

pub const FUNC_F: f64 = 1.0;

pub const ALPHA: f64 = 1.0;
pub const BETA: f64 = -1.0;

fn boundary(node_num_glo: usize, dirichlet: f64, neumann: f64,
            mat_a_glo: &mut Array2<f64>, vec_b_glo: &mut Array1<f64>) 
{
    if dirichlet != INFINITY {
        Zip::from(vec_b_glo.slice_mut(s![..]))
            .and(mat_a_glo.slice(s![node_num_glo, ..]))
            .apply(|b, &a| *b -= dirichlet * a);
        vec_b_glo[node_num_glo] = dirichlet;
        Zip::from(&mut mat_a_glo.slice_mut(s![node_num_glo, ..]))
            .apply(|a| *a = 0.0);
        Zip::from(&mut mat_a_glo.slice_mut(s![.., node_num_glo]))
            .apply(|a| *a = 0.0);
        mat_a_glo[[node_num_glo, node_num_glo]] = 1.0;
    }

    if neumann != INFINITY {
        vec_b_glo[node_num_glo] += neumann;
    }
}

fn main() {
    println!("node_total = {}, ele_total = {}\n", NODE_TOTAL, ELE_TOTAL);
    println!("Global node X");

    let node_x_glo = Array::linspace(X_MIN, X_MAX, NODE_TOTAL);
    println!("{:?}\n", node_x_glo);

    println!("Node number of each elements");
    let mut node_num_glo_in_seg_ele: Array2<usize> = Array::zeros((ELE_TOTAL, 2));
    for e in 0..ELE_TOTAL {
        node_num_glo_in_seg_ele[[e, 0]] = e;
        node_num_glo_in_seg_ele[[e, 1]] = e+1;
    }
    println!("{:?}\n", node_num_glo_in_seg_ele);

    println!("Local node X");
    let mut node_x_ele: Array2<f64> = Array::zeros((ELE_TOTAL, 2));
    for e in 0..ELE_TOTAL {
        for i in 0..2 {
            node_x_ele[[e, i]] = node_x_glo[ node_num_glo_in_seg_ele[[e, i]] ];
        }
    }
    println!("{:?}\n", node_x_ele);

    println!("Element length");
    let mut length: Array1<f64> = Array::zeros(ELE_TOTAL);
    for e in 0..ELE_TOTAL {
        length[e] = (node_x_ele[[e, 1]] - node_x_ele[[e, 0]]).abs();
    }
    println!("{:?}\n", length);

    let mut mat_a_ele: Array3<f64> = Array::zeros((ELE_TOTAL, 3, 3));
    let mut vec_b_ele: Array2<f64> = Array::zeros((ELE_TOTAL, 3));

    //println!("Local matrix");
    for e in 0..ELE_TOTAL {
        for i in 0..2 {
            for j in 0..2 {
                mat_a_ele[[e, i, j]] = (-1.0).powi(i as i32 + 1) * (-1.0).powi(j as i32 + 1) / length[e];
            }
            vec_b_ele[[e, i]] = -FUNC_F * length[e] / 2.0;
        }
    }

    let mut mat_a_glo: Array2<f64> = Array::zeros((NODE_TOTAL, NODE_TOTAL));
    let mut vec_b_glo: Array1<f64> = Array::zeros(NODE_TOTAL);

    println!("Global matrix (constructed)");
    for e in 0..ELE_TOTAL {
        for i in 0..2 {
            for j in 0..2 {
                mat_a_glo[[node_num_glo_in_seg_ele[[e, i]], node_num_glo_in_seg_ele[[e, j]]]] += mat_a_ele[[e, i, j]];
            }
            vec_b_glo[node_num_glo_in_seg_ele[[e, i]]] += vec_b_ele[[e, i]];
        }
    }

    if NODE_TOTAL < 20 {
        for i in 0..NODE_TOTAL {
            for j in 0..NODE_TOTAL {
                print!("{:6.2} ", mat_a_glo[[i, j]]);
            }
            println!(";{:6.2}", vec_b_glo[i]);
        }
    }
    println!();

    boundary(0, ALPHA, 0.0, &mut mat_a_glo, &mut vec_b_glo);
    boundary(NODE_TOTAL-1, INFINITY, BETA, &mut mat_a_glo, &mut vec_b_glo);

    println!("Post global matrix");
    if NODE_TOTAL < 20 {
        for i in 0..NODE_TOTAL {
            for j in 0..NODE_TOTAL {
                print!("{:6.2} ", mat_a_glo[[i, j]]);
            }
            println!(";{:6.2}", vec_b_glo[i]);
        }
    }
    println!();

    println!("node_total = {}, ele_total = {}\n", NODE_TOTAL, ELE_TOTAL);
    println!("Unknown vector u = ");
    let unknown_vec_u = mat_a_glo.solve(&vec_b_glo).unwrap();
    println!("{:?}\n", unknown_vec_u);

    let (mut ma, mut mi) = (-INFINITY, INFINITY);
    for &u in unknown_vec_u.iter() {
        if u > ma { ma = u }
        if u < mi { mi = u }
    }
    println!("Max u = {}, Min u = {}", ma, mi);

    let exact_x: Array1<f64> = Array::linspace(X_MIN, X_MAX, 1000);
    let exact_y: Array1<f64> = Array::from(
                                    exact_x.iter()
                                           .map(|&x| FUNC_F/2.0*x.powi(2)
                                                    +(-FUNC_F*X_MAX+BETA)*x
                                                    -FUNC_F/2.0*X_MIN.powi(2)
                                                    -(-FUNC_F*X_MAX+BETA)*X_MIN
                                                    +ALPHA)
                                           .collect::<Vec<f64>>()
                                );

    let mut fg = gnuplot::Figure::new();
    fg.axes2d().lines(exact_x.iter(), exact_y.iter(), &[gnuplot::Color("red"), gnuplot::Caption("u(x)")])
               .lines(node_x_glo.iter(), unknown_vec_u.iter(), &[gnuplot::Color("blue"), gnuplot::Caption("u*(x)")])
               .points(node_x_glo.iter(), unknown_vec_u.iter(), &[gnuplot::Color("black")])
               .set_title("fem1d_poisson", &[])
               .set_x_label("x", &[])
               .set_y_label("y", &[]);
    fg.echo_to_file("fem1d_poisson.plt");
}