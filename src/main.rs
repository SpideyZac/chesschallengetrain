pub mod game;
pub mod genetic;
pub mod population;

use crate::game::SpikingCellularNN;

fn main() {
    const WIDTH: usize = 10;
    const HEIGHT: usize = 10;
    const TIME_STEPS: usize = 10;

    let mut network = SpikingCellularNN::new(WIDTH, HEIGHT);

    network.printgame();

    for _ in 0..TIME_STEPS {
        network.update_cells();

        network.printgame();
    }
}
