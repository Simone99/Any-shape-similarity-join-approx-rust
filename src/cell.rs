use std::{cmp::Ordering, collections::HashMap, fmt, hash::Hash};

use crate::point::Point;

#[derive(Clone)]
pub struct Cell {
    pub points_set: HashMap<u32, Vec<Point>>,
    pub m: Vec<u64>,
    #[cfg(feature = "normal")]
    pub m_c: u64,
    pub coordinates: Vec<i32>,
    #[cfg(feature = "cell_distance_vertices")]
    pub cell_vertices: Vec<Point>,
    #[cfg(feature = "cell_distance_center")]
    pub cell_center: Point,
    #[cfg(feature = "cell_distance_center")]
    pub eps: f32,
}

impl Cell {
    pub fn distance_from(&self, other: &Cell) -> f32 {
        #[cfg(not(feature = "cell_distance_center"))]
        let mut result: f32 = f32::MAX;
        #[cfg(feature = "cell_distance_points")]
        {
            for set_a in &self.points_set {
                for set_b in &other.points_set {
                    for a in set_a.1 {
                        for b in set_b.1 {
                            let distance = a.distance_from(&b);
                            if distance < result {
                                result = distance;
                            }
                        }
                    }
                }
            }
            return result;
        }
        #[cfg(feature = "cell_distance_vertices")]
        {
            for a in &self.cell_vertices {
                for b in &other.cell_vertices {
                    let distance = a.distance_from(&b);
                    if distance < result {
                        result = distance;
                    }
                }
            }
            return result;
        }
        #[cfg(feature = "cell_distance_center")]
        {
            #[cfg(feature = "l1_metric")]
            // Somehow it doesn't work
            return self.cell_center.distance_from(&other.cell_center) - self.eps;
            #[cfg(feature = "l_inf_metric")]
            // Somehow it doesn't work
            return self.cell_center.distance_from(&other.cell_center) - self.eps;
            #[cfg(feature = "l2_metric")]
            return self.cell_center.distance_from(&other.cell_center)
                - (self.cell_center.coordinates.len() as f32).sqrt() * self.eps;
        }
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.coordinates.cmp(&other.coordinates)
    }
}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.coordinates.cmp(&other.coordinates))
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.coordinates == other.coordinates
    }
}

impl Eq for Cell {}

impl Hash for Cell {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coordinates.hash(state);
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cell: (").expect("Error writing a point to the output stream!");
        for i in 0..(self.coordinates.len() - 1) {
            write!(f, "{}, ", self.coordinates[i])
                .expect("Error writing a point to the output stream!");
        }
        write!(f, "{})", self.coordinates[self.coordinates.len() - 1])
    }
}
