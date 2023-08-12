use rand::{thread_rng, Rng};

use crate::game::SpikingCellularNN;

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
        width: usize,
        height: usize,
        mutation_rate: f64,
        inputs: Vec<Point>,
        outputs: Vec<Point>,
        activator: Point,
    ) -> Self {
        let random = thread_rng();

        GeneticAlgorithm {
            nn: SpikingCellularNN::new(width, height),
            mutation_rate,
            inputs,
            outputs,
            activator,
            random,
        }
    }

    pub fn mutate(&mut self) {
        for y in 0..self.nn.height {
            for x in 0..self.nn.width {
                if self.random.gen_range(0.0..1.0) < self.mutation_rate {
                    self.nn.start_cells[y][x].threshold += self.random.gen_range(-1.0..1.0);
                    self.nn.start_cells[y][x].activation += self.random.gen_range(-1.0..1.0);
                }
            }
        }
    }

    pub fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        let mut outputs = vec![];
        self.nn.reset();

        for (i, input) in inputs.iter().enumerate() {
            let input_point = &self.inputs[i];
            self.nn.cells[input_point.row][input_point.col].activation = *input;
        }

        // 100 max iter
        for _ in 0..100 {
            if self.nn.cells[self.activator.row][self.activator.col].spiked {
                for output_point in self.outputs.iter() {
                    outputs.push(self.nn.cells[output_point.row][output_point.col].activation);
                }
                break;
            }

            self.nn.update_cells();
        }

        outputs.resize(self.outputs.len(), 0.0);

        outputs
    }
}
