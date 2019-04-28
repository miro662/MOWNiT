#[macro_use]
extern crate itertools;

use std::ops::{Add};
use std::mem::swap;
use rand::Rng;

#[derive(Debug, Copy, Clone)]
struct Point(isize, isize);

impl Add for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        let Point(x, y) = self;
        let Point(xx, yy) = rhs;
        Point(x + xx, y + yy)
    }
}

impl Point {
    fn random(size: Point) -> Point {
        let Point(x, y) = size;
        let mut rng = rand::thread_rng();
        Point(rng.gen_range(0, x), rng.gen_range(0, y))
    }
}

fn sqaure_neighbourhood(of: Point) -> Vec<Point> {
    let offsets = vec!{
        Point(-1, 0),
        //Point(1, 0),
        //Point(0, -1),
        //Point(0, 1),
    };

    let offsets_mapped = offsets.iter().map(
        |x| *x + of
    );

    offsets_mapped.collect()
}

fn energy(neigbourhood: impl Iterator<Item = bool>) -> u32 {
    neigbourhood.filter(|x| *x).count() as u32
}

trait Anneable {
    fn get_neighbour(&self) -> Self;
    fn get_energy(&self) -> f64;
}

#[derive(Clone)]
struct Map {
    size: Point,
    data: Vec<bool>
}

impl Map {
    fn generate(size: Point, probability: f32) -> Map {
        let Point(x, y) = size;
        let mut rng = rand::thread_rng();

        Map {
            size: size,
            data: (0..(x*y)).map(|_| rng.gen_range::<f32, f32, f32>(0.0, 1.0) < probability).collect()
        }
    }

    fn wrap_point(&self, point: Point) -> Point {
        let Point(mut x, mut y) = point;
        let Point(xsize, ysize) = self.size;

        if x < 0 {
            x += xsize;
        }
        if y < 0 {
            y += ysize;
        }

        Point(x % xsize, y % ysize)
    }
    
    fn wrap_points<'a>(&'a self, points: impl Iterator<Item = &'a Point> + 'a) -> impl Iterator<Item = Point> + 'a {
        points.map(move |point| self.wrap_point(*point))
    }

    fn to_index(&self, pos: Point) -> usize {
        let Point(x, y) = pos;
        let Point(xsize, _) = self.size;

        (y * xsize + x) as usize
    }

    fn get_energy_data<'a>(&'a self, neigbourhood: impl Iterator<Item = Point> + 'a) -> impl Iterator<Item = bool> + 'a {
        neigbourhood.map(move |pos| self.data[self.to_index(pos)])
    }

    fn get_map_energy(&self) -> u32 {
        let Point(x, y) = self.size;
        iproduct!(0..x, 0..y).map(|(xx, yy)| {
            let neighbourhood = sqaure_neighbourhood(Point(xx, yy));
            let wrapped_neighbourhood = self.wrap_points(neighbourhood.iter());
            let points_data = self.get_energy_data(wrapped_neighbourhood);
            energy(points_data)
        }).sum()
    }

    fn random_swap(&self) -> Self {
        let mut swapped = self.clone();
        let (a_point, b_point) = (Point::random(self.size), Point::random(self.size));
        let (a, b) = (self.to_index(a_point), self.to_index(b_point));
        swapped.data[a] = self.data[b];
        swapped.data[b] = self.data[a];
        swapped
    }
}

impl Anneable for Map {
    fn get_neighbour(&self) -> Map {
        self.random_swap()
    }

    fn get_energy(&self) -> f64 {
        self.get_map_energy().into()
    }
}

fn metropolis(previous_energy: f64, new_energy: f64, temperature: f64) -> bool {
    if new_energy < previous_energy {
        true
    } else {
        let mut rng = rand::thread_rng();
        let random: f64 = rng.gen();
        let probability = ((previous_energy - new_energy) / temperature).exp();
        random > probability
    }
}

fn simulated_annealing<T>(initial_state: T, iterations: usize, initial_temperature: f64, temperature_func: impl Fn(f64)->f64) -> T
where T: Anneable {
    let mut state = initial_state;
    let mut energy = state.get_energy();
    let mut t = initial_temperature;

    for iteration in 0..iterations {
        let neigbour = state.get_neighbour();
        let neigbour_energy = neigbour.get_energy();
        if metropolis(energy, neigbour_energy, t) {
            state = neigbour;
            energy = neigbour_energy;
        }

        t = temperature_func(t);
    }
    state
}

fn main() {
    let map = Map::generate(Point(16, 16), 0.5);
    let final_state = simulated_annealing(map, 100000, 100.0, |t| t * 0.999);
    println!("{:?}", final_state.data);
}
