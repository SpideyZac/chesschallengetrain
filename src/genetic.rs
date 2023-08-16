use rand::{thread_rng, Rng};
use std::fs::File;
use std::io::{Read, Write};

use crate::{game::{SpikingCellularNN, Cell}, HEIGHT, WIDTH};

#[derive(Clone, Copy)]
pub struct Point {
    pub row: usize,
    pub col: usize,
}

#[derive(Clone)]
pub struct GeneticAlgorithm {
    pub nn: SpikingCellularNN,
    pub mutation_rate: f64,
    pub inputs: Vec<Point>,
    pub outputs: Vec<Point>,
    pub activator: Point,
    pub random: rand::rngs::ThreadRng,
}

impl GeneticAlgorithm {
    pub fn new(
        mutation_rate: f64,
        inputs: Vec<Point>,
        outputs: Vec<Point>,
        activator: Point,
    ) -> Self {
        let random = thread_rng();

        GeneticAlgorithm {
            nn: SpikingCellularNN::default(),
            mutation_rate,
            inputs,
            outputs,
            activator,
            random,
        }
    }

    pub fn mutate(&mut self) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                if self.random.gen_range(0.0..1.0) < self.mutation_rate {
                    self.nn.start_cells[y][x].threshold += self.random.gen_range(-1.0..1.0);
                    self.nn.start_cells[y][x].activation += self.random.gen_range(-1.0..1.0);
                }
            }
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
        mutation_rate: f64,
        inputs: Vec<Point>,
        outputs: Vec<Point>,
        activator: Point,
    ) -> Self {
        let mut ga = GeneticAlgorithm::new(mutation_rate, inputs, outputs, activator);
        let mut file = File::open(filename).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        for (i, line) in contents.lines().enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let activation = parts[0].parse::<f64>().unwrap();
            let spiked = parts[1].parse::<i32>().unwrap() != 0;
            let threshold = parts[2].parse::<f64>().unwrap();

            ga.nn.start_cells[i / WIDTH][i % WIDTH] = Cell {
                activation,
                spiked,
                threshold,
            };
        }

        ga
    }
}
