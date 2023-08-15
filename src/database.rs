use std::fs::File;
use rand::Rng;
use csv::ReaderBuilder;

#[derive(Debug)]
pub struct ChessSample {
    fen: String,
    evaluation: i32,
}

pub fn read_csv(file_path: &str) -> Result<Vec<ChessSample>, csv::Error> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().from_reader(file);
    let mut samples = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let fen = record.get(0).unwrap().to_string();
        let evaluation = if record.get(1).unwrap().chars().nth(0).unwrap() == '#' {
            10000
        } else {
            record.get(1).unwrap().parse::<i32>().unwrap()
        };

        samples.push(ChessSample { fen, evaluation });
    }

    Ok(samples)
}

pub fn generate_random_batch(samples: &[ChessSample], batch_size: usize) -> Vec<&ChessSample> {
    let mut rng = rand::thread_rng();
    let mut batch = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        let random_index = rng.gen_range(0..samples.len());
        batch.push(&samples[random_index]);
    }

    batch
}

fn fen_to_board_vector(fen: &str) -> (bool, Vec<i8>) {
    let mut board_vector = vec![0; 64];

    let mut rank = 7;
    let mut file = 0;
    let mut is_white_to_move = true; // Default to white's turn

    for c in fen.chars() {
        match c {
            '1'..='8' => {
                let empty_squares = c.to_digit(10).unwrap() as usize;
                file += empty_squares;
            }
            '/' => {
                rank -= 1;
                file = 0;
            }
            'w' => {
                is_white_to_move = true;
            }
            'b' => {
                is_white_to_move = false;
            }
            _ => {
                let piece_value = match c {
                    'p' => -1,
                    'n' => -2,
                    'b' => -3,
                    'r' => -4,
                    'q' => -5,
                    'k' => -6,
                    'P' => 1,
                    'N' => 2,
                    'B' => 3,
                    'R' => 4,
                    'Q' => 5,
                    'K' => 6,
                    _ => 0,
                };
                board_vector[rank * 8 + file] = piece_value;
                file += 1;
            }
        }
    }

    (is_white_to_move, board_vector)
}

pub fn process_batch_for_training(samples: &[&ChessSample]) -> (Vec<Vec<i8>>, Vec<i32>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for sample in samples {
        let (turn, mut board_vector) = fen_to_board_vector(&sample.fen);
        let mut evaluation = sample.evaluation;

        if !turn {
            for val in &mut board_vector {
                *val *= -1;
            }
        }

        if turn {
            evaluation *= -1;
        }

        inputs.push(board_vector);
        targets.push(evaluation);
    }

    (inputs, targets)
}
