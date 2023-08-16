use std::fs::File;
use std::io::{Read, Write};

use crate::game::{SpikingCellularNN, Cell};
use crate::genetic::Point;

#[derive(Clone)]
pub struct Model {
    pub nn: SpikingCellularNN,
    pub inputs: Vec<Point>,
    pub outputs: Vec<Point>,
    pub activator: Point,
}

impl Model {
    pub fn new(
        width: usize,
        height: usize,
        inputs: Vec<Point>,
        outputs: Vec<Point>,
        activator: Point,
    ) -> Self {
        Model {
            nn: SpikingCellularNN::new(width, height),
            inputs,
            outputs,
            activator,
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> (Vec<f64>, u8) {
        let mut outputs = Vec::with_capacity(self.outputs.len());
        self.nn.reset();

        for (i, input) in inputs.iter().enumerate() {
            let input_point = &self.inputs[i];
            self.nn.cells[input_point.row][input_point.col].activation = *input;
        }

        // 100 max iter
        let mut i = 0;
        for _ in 0..100 {
            if self.nn.cells[self.activator.row][self.activator.col].spiked {
                for output_point in self.outputs.iter() {
                    outputs.push(self.nn.cells[output_point.row][output_point.col].activation);
                }
                break;
            }

            self.nn.update_cells();

            i += 1;
        }

        if outputs.len() != self.outputs.len() {
            for output_point in self.outputs.iter() {
                outputs.push(self.nn.cells[output_point.row][output_point.col].activation);
            }
        }

        (outputs, i)
    }

    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(filename)?;

        for row in &self.nn.start_cells {
            for cell in row {
                let serialized_cell = format!(
                    "{:.8} {:.8} {:.8}\n",
                    cell.activation, cell.spiked as i32, cell.threshold
                );
                file.write_all(serialized_cell.as_bytes())?;
            }
        }

        Ok(())
    }

    pub fn load(
        filename: &str,
        width: usize,
        height: usize,
        inputs: Vec<Point>,
        outputs: Vec<Point>,
        activator: Point,
    ) -> Self {
        let mut ga = Model::new(width, height, inputs, outputs, activator);
        let mut file = File::open(filename).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let mut start_cells = vec![];
        for line in contents.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let activation = parts[0].parse::<f64>().unwrap();
            let spiked = parts[1].parse::<i32>().unwrap() != 0;
            let threshold = parts[2].parse::<f64>().unwrap();

            let cell = Cell {
                activation,
                spiked,
                threshold,
            };

            start_cells.push(cell);
        }

        let width = width; // Set the original width here
        let height = height; // Set the original height here
        let mut start_cells_2d = vec![];
        for _ in 0..height {
            let row = start_cells.drain(..width).collect::<Vec<_>>();
            start_cells_2d.push(row);
        }

        ga.nn.start_cells = start_cells_2d;

        ga
    }

    pub fn reward(&mut self, target_inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
        let mut reward = 0.0;

        for i in 0..target_inputs.len() {
            let (outputs, iters) = self.forward(target_inputs[i].as_slice());
            for (j, output) in outputs.iter().enumerate() {
                reward -= (output - targets[i][j]).abs();
            }
            reward -= iters as f64 / 10.0;
        }

        reward / target_inputs.len() as f64
    }

    pub fn gradient_ascent(&mut self, iterations: u16, learning_rate: f64, target_inputs: Vec<Vec<f64>>, target_outputs: Vec<Vec<f64>>) {
        let epsilon = 1e-8;
        let mut prev = -1000.0;

        for iter in 0..iterations {
            let mut gradients_activation: Vec<f64> = Vec::with_capacity(self.nn.width * self.nn.height);
            let mut gradients_threshold: Vec<f64> = Vec::with_capacity(self.nn.width * self.nn.height);
            let orig_reward = self.reward(&target_inputs, &target_outputs);

            for i in 0..self.nn.start_cells.len() {
                for j in 0..self.nn.start_cells[i].len() {
                    self.nn.start_cells[i][j].activation += epsilon;
                    let reward = self.reward(&target_inputs, &target_outputs);
                    self.nn.start_cells[i][j].activation -= epsilon;
                    gradients_activation.push((reward - orig_reward) / epsilon * learning_rate);
                    self.nn.start_cells[i][j].threshold += epsilon;
                    let reward = self.reward(&target_inputs, &target_outputs);
                    self.nn.start_cells[i][j].threshold -= epsilon;
                    gradients_threshold.push((reward - orig_reward) / epsilon * learning_rate);
                }
            }

            for i in 0..self.nn.start_cells.len() {
                for j in 0..self.nn.start_cells[i].len() {
                    self.nn.start_cells[i][j].activation += gradients_activation[i * self.nn.start_cells[i].len() + j];
                    self.nn.start_cells[i][j].threshold += gradients_threshold[i * self.nn.start_cells[i].len() + j];
                }
            }

            println!("Iteration: {} Reward: {:.5} Better: {}", iter, self.reward(&target_inputs, &target_outputs), self.reward(&target_inputs, &target_outputs) > prev);
            prev = self.reward(&target_inputs, &target_outputs);
        }
    }
}
