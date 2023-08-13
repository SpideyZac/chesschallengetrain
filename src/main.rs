pub mod game;
pub mod genetic;
pub mod population;

use crate::population::Population;
use crate::genetic::Point;

fn main() {
    const WIDTH: usize = 5;
    const HEIGHT: usize = 5;

    let mut pop = Population::new(
        100,
        WIDTH,
        HEIGHT,
        vec![
            Point {
                row: 1,
                col: 0,
            },
            Point {
                row: 3,
                col: 0,
            },
        ],
        vec![
            Point {
                row: 2,
                col: 4,
            },
        ],
        Point {
            row: 0,
            col: 4,
        },
        0.75,
    );

    let train = vec![
        vec![vec![0.0, 0.0], vec![0.0]],
        vec![vec![0.0, 1.0], vec![1.0]],
        vec![vec![1.0, 1.0], vec![0.0]],
        vec![vec![1.0, 0.0], vec![1.0]],
    ];

    let mut best = -1000.0;
    let mut b = pop.population[0].clone();

    let mut stale = 0;
    let mut last_best = 0.0;

    for iter in 0..10000 {
        if iter % 100 == 0 {
            println!("Iteration: {} Best: {:.5}", iter, best);

            if best == last_best {
                stale += 1;
            } else {
                stale = 0;
                last_best = best;
            }

            if stale == 10 {
                println!("Stale Level 1!");

                for i in 0..pop.population.len() {
                    pop.population[i] = b.clone();
                }
            } else if stale == 20 {
                println!("Stale Level 2... Reseting!");
                pop = Population::new(
                    100,
                    WIDTH,
                    HEIGHT,
                    vec![
                        Point {
                            row: 1,
                            col: 0,
                        },
                        Point {
                            row: 3,
                            col: 0,
                        },
                    ],
                    vec![
                        Point {
                            row: 2,
                            col: 4,
                        },
                    ],
                    Point {
                        row: 0,
                        col: 4,
                    },
                    0.75,
                );
            }
        }

        if best >= 4.0 {
            break;
        }

        let mut rewards = vec![];

        for ga in pop.population.iter_mut() {
            let mut reward = 0.0;

            for data_point in train.iter() {
                let (output, _) = ga.forward(data_point[0].as_slice());
                if ga.nn.cells[ga.activator.row][ga.activator.col].spiked {
                    reward += 1.0;
                }
                reward += -((output[0] - data_point[1][0]).abs());
            }

            rewards.push(reward);
        }

        for (i, reward) in rewards.iter().enumerate() {
            pop.add_rew(*reward);
            if *reward > best {
                best = *reward;
                b = pop.population[i].clone();
            }
        }

        pop.learn();
    }
}
