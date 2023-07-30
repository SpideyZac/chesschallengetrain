use crate::genetic::{GeneticAlgorithm, Point};

pub struct Population {
    pub population: Vec<GeneticAlgorithm>,
}

impl Population {
    pub fn new(
        size: u32,
        width: usize,
        height: usize,
        input_points: Vec<Point>,
        output_points: Vec<Point>,
        activator_point: Point,
        mutation_rate: f64,
    ) -> Self {
        let mut population = vec![];

        for _ in 0..size {
            population.push(GeneticAlgorithm::new(
                width,
                height,
                mutation_rate,
                input_points.clone(),
                output_points.clone(),
                activator_point,
            ));
        }

        Population { population }
    }
}
