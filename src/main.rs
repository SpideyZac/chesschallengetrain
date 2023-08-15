pub mod game;
pub mod genetic;
pub mod population;
pub mod database;

use crate::genetic::Point;
use crate::population::Population;
use crate::database::{read_csv, generate_random_batch, process_batch_for_training};

fn main() {
    let samples_file = read_csv("C:\\Users\\zacle\\Downloads\\archive\\chessData.csv").unwrap();
    let samples = samples_file.as_slice();

    let size = 100;
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
    let mutation_rate = 0.5;

    let mut pop = Population::new(
        size,
        width,
        height,
        input_points,
        output_points,
        activator_point,
        mutation_rate
    );

    let mut best = -100000.0;

    for iter in 0..1000 {
        println!("Iter: {} Best: {:.5}", iter, best);
        let (target_inputs, target_outputs) = process_batch_for_training(generate_random_batch(samples, 16).as_slice());

        
        for ga in pop.population.iter_mut() {
            let mut reward = 0.0;

            for i in 0..16 {
                let (output, iters) = ga.forward(target_inputs[i].iter().map(|&x| x as f64).collect::<Vec<f64>>().as_slice());
            
                reward -= (output[0] - target_outputs[i] as f64).abs();
                reward -= iters as f64;
            }

            if reward > best {
                best = reward;
            }

            pop.rewards.push(reward);
        }

        pop.learn();
    }
}
