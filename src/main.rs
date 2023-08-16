pub mod game;
pub mod genetic;
pub mod population;
pub mod database;
pub mod model;

use crate::genetic::Point;
// use crate::population::Population;
use crate::database::{read_csv, generate_random_batch, process_batch_for_training};
use crate::model::Model;

// use std::collections::VecDeque;

fn main() {
    let samples_file = read_csv("C:\\Users\\zacle\\Downloads\\archive\\chessData.csv").unwrap();
    let samples = samples_file.as_slice();

    // let size = 100;
    let width = 20;
    let height = 16;
    let mut input_points = vec![];
    for i in 0..8 {
        for j in 0..8 {
            input_points.push(Point {
                row: i * 2,
                col: j * 2,
            });
        }
    }
    let output_points = vec![Point {
        row: 7,
        col: 19,
    }];
    let activator_point = Point {
        row: 0,
        col: 19,
    };
    // let mutation_rate = 0.5;

    // let mut pop = Population::new(
    //     size,
    //     width,
    //     height,
    //     input_points,
    //     output_points,
    //     activator_point,
    //     mutation_rate
    // );

    // let mut best = -100000.0;
    // let mut b = pop.population[0].clone();
    // let mut testbatchesinput: VecDeque<Vec<Vec<i8>>> = VecDeque::with_capacity(10);
    // let mut testbatchesoutputs: VecDeque<Vec<i32>> = VecDeque::with_capacity(10);
    // let mut testbatchesreward = -100000.0;

    // for iter in 0..1000 {
    //     if iter % 10 == 0 {
    //         println!("Iter: {} Best: {:.5}", iter, best);
    //     }
    //     let (target_inputs, target_outputs) = process_batch_for_training(generate_random_batch(samples, 16).as_slice());

        
    //     for ga in pop.population.iter_mut() {
    //         let mut reward = 0.0;

    //         for i in 0..16 {
    //             let (output, iters) = ga.forward(target_inputs[i].iter().map(|&x| x as f64).collect::<Vec<f64>>().as_slice());
            
    //             reward -= (output[0] - target_outputs[i] as f64).abs();
    //             reward -= (iters as f64) / 10.0;
    //         }

    //         if reward > best {
    //             let mut r = 0.0;

    //             if testbatchesinput.len() != 0 {
    //                 for i in 0..testbatchesinput.len() {
    //                     for j in 0..testbatchesinput[i].len() {
    //                         let (output, iters) = ga.forward(testbatchesinput[i][j].iter().map(|&x| x as f64).collect::<Vec<f64>>().as_slice());
                    
    //                         r -= (output[0] - testbatchesoutputs[i][j] as f64).abs();
    //                         r -= (iters as f64) / 10.0;
    //                     }
    //                 }

    //                 r /= testbatchesinput.len() as f64;
    //             } else {
    //                 r = reward;
    //             }

    //             if r > testbatchesreward {
    //                 if testbatchesinput.binary_search(&target_inputs).is_err() {
    //                     testbatchesinput.push_front(target_inputs.clone());
    //                     testbatchesoutputs.push_front(target_outputs.clone());
    //                 }

    //                 best = reward;
    //                 b = ga.clone();
    //                 testbatchesreward = r;
    //             }
    //         }

    //         pop.rewards.push(reward);
    //     }

    //     pop.learn();
    //     pop.population[99] = b.clone();
    // }

    // b.save("best").unwrap();

    let mut model = Model::new(width, height, input_points, output_points, activator_point);
    for iter in 0..100 {
        let batch = process_batch_for_training(generate_random_batch(samples, 16).as_slice());
        let mut target_outputs = vec![];
        for output in batch.1.iter() {
            target_outputs.push(vec![*output as f64]);
        }
        println!("Batch: {}", iter);
        model.gradient_ascent(5, 0.001, batch.0.iter().map(|x| x.iter().map(|&y| y as f64).collect()).collect::<Vec<Vec<f64>>>(), target_outputs);
    }
    model.save("model").unwrap();
}
