use rand::{Rng, thread_rng};

#[derive(Clone)]
pub struct Cell {
    pub activation: f64,
    pub spiked: bool,
    pub threshold: f64,
}

pub struct SpikingCellularNN {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<Cell>>,
    pub start_cells: Vec<Vec<Cell>>,
}

impl SpikingCellularNN {
    pub fn new(width: usize, height: usize) -> Self {
        let mut random = thread_rng();

        let mut cells = vec![vec![]];

        for i in 0..height {
            cells.push(vec![]);
            for _ in 0..width {
                cells[i].push(Cell { activation: random.gen_range(0f64..1f64), spiked: false, threshold: random.gen_range(0f64..1f64) });
            }
        }

        let start_cells = cells.clone();

        SpikingCellularNN {
            width,
            height,
            cells,
            start_cells,
        }
    }

    pub fn update_cells(&mut self) {
        let mut new_cells = self.cells.clone();

        for y in 0..self.height {
            for x in 0..self.width {
                let mut new_activation = self.cells[y][x].activation;

                let mut num_neighbors = 0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;

                        if nx >= 0 && nx < self.width as isize && ny >= 0 && ny < self.height as isize {
                            new_activation += self.cells[ny as usize][nx as usize].activation;
                            num_neighbors += 1;
                        }
                    }
                }

                if num_neighbors > 0 {
                    new_activation /= num_neighbors as f64;
                }

                if new_activation > self.cells[y][x].threshold {
                    new_cells[y][x].spiked = true;
                    new_activation = 0.0;
                } else {
                    new_cells[y][x].spiked = false;
                }

                new_cells[y][x].activation = new_activation;
            }
        }

        self.cells = new_cells;
    }

    pub fn printgame(&self) {
        for y in 0..self.height {
            for x in 0..self.width {
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
        self.cells = self.start_cells.clone();
    }
}