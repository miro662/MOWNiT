#[macro_use]
extern crate itertools;

use std::ops::{Add};
use rand::Rng;
use rand::distributions::uniform::Uniform;
use std::iter::Product;

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

fn sqaure_neighbourhood(of: Point) -> Vec<Point> {
    let offsets = vec!{
        Point(-1, 0),
        Point(1, 0),
        Point(0, -1),
        Point(0, 1),
    };

    let offsets_mapped = offsets.iter().map(
        |x| *x + of
    );

    offsets_mapped.collect()
}

fn energy(neigbourhood: impl Iterator<Item = bool>) -> u32 {
    neigbourhood.filter(|x| *x).count() as u32
}

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
            let neighbourhood = sqaure_neighbourhood(Point(x, y));
            let wrapped_neighbourhood = self.wrap_points(neighbourhood.iter());
            let points_data = self.get_energy_data(wrapped_neighbourhood);
            energy(points_data)
        }).sum()
    }
}

fn main() {
    let map = Map::generate(Point(512, 512), 0.75);
    println!("{}", map.get_map_energy());
}
