use crate::genetic::{GeneticAlgorithm, Point};

#[derive(Clone)]
pub struct Population {
    pub population: Vec<GeneticAlgorithm>,
    pub rewards: Vec<f64>,
}

impl Population {
    pub fn new(
        size: u32,
        input_points: Vec<Point>,
        output_points: Vec<Point>,
        activator_point: Point,
        mutation_rate: f64,
    ) -> Self {
        let mut population = Vec::with_capacity(size as usize);

        for _ in 0..size {
            population.push(GeneticAlgorithm::new(
                mutation_rate,
                input_points.clone(),
                output_points.clone(),
                activator_point,
            ));
        }

        Population {
            population,
            rewards: Vec::with_capacity(size as usize),
        }
    }

    pub fn learn(&mut self) {
        let mut best_rew = 0.0;
        let mut best = self.population[0].clone();

        for (i, reward) in self.rewards.iter().enumerate() {
            if *reward > best_rew {
                best_rew = *reward;
                best = self.population[i].clone();
            }
        }

        for ga in self.population.iter_mut() {
            let mut new = best.clone();
            new.mutate();
            *ga = new;
        }

        self.rewards = Vec::with_capacity(self.population.len());
    }

    pub fn add_rew(&mut self, reward: f64) {
        self.rewards.push(reward);
    }
}
