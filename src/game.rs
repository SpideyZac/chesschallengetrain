use rand::{Rng, thread_rng};
use crate::{WIDTH, HEIGHT};

#[derive(Clone, Copy, Default)]
pub struct Cell {
    pub activation: f64,
    pub spiked: bool,
    pub threshold: f64,
}

#[derive(Clone)]
pub struct SpikingCellularNN {
    pub cells: [[Cell; WIDTH]; HEIGHT],
    pub start_cells: [[Cell; WIDTH]; HEIGHT],
}

impl Default for SpikingCellularNN {
    fn default() -> Self {
        let mut random = thread_rng();

        let mut res = Self {
            cells: [[Cell::default(); WIDTH]; HEIGHT],
            start_cells: [[Cell::default(); WIDTH]; HEIGHT],
        };

        for i in 0..HEIGHT {
            for j in 0..WIDTH {
                let cell = Cell {
                    activation: random.gen_range(0.0..1.0),
                    spiked: false,
                    threshold: random.gen_range(0.0..1.0),
                };

                res.cells[i][j] = cell;
                res.start_cells[i][j] = cell;
            }
        }

        res
    }
}

impl SpikingCellularNN {
    pub fn update_cells(&mut self) {
        let mut new_cells = self.cells;

        for (y, new_row) in new_cells.iter_mut().enumerate() {
            for (x, new_cell) in new_row.iter_mut().enumerate() {
                let mut new_activation = self.cells[y][x].activation;

                let mut num_neighbors = 0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;

                        if nx >= 0 && nx < WIDTH as isize && ny >= 0 && ny < HEIGHT as isize {
                            new_activation += self.cells[ny as usize][nx as usize].activation;
                            num_neighbors += 1;
                        }
                    }
                }

                if num_neighbors > 0 {
                    new_activation /= num_neighbors as f64;
                }

                if new_activation > self.cells[y][x].threshold {
                    new_cell.spiked = true;
                    new_activation = 0.0;
                } else {
                    new_cell.spiked = false;
                }

                new_cell.activation = new_activation;
            }
        }

        self.cells = new_cells;
    }

    pub fn printgame(&self) {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let cell = &self.cells[y][x];
                if cell.spiked {
                    print!("X{:.1} ", cell.activation);
                } else {
                    print!("{:.2} ", cell.activation);
                }
            }
            println!();
        }
        println!();
    }

    pub fn reset(&mut self) {
        self.cells = self.start_cells;
    }
}