use std::fmt;

#[derive(PartialEq, PartialOrd, Clone)]
pub struct Point {
    pub coordinates: Vec<f32>,
}

impl Point {
    pub fn new(coordinates: &Vec<f32>) -> Point {
        Point {
            coordinates: coordinates.clone(),
        }
    }

    pub fn distance_from(&self, other: &Point) -> f32 {
        #[cfg(feature = "l1_metric")]
        return (0..self.coordinates.len())
            .map(|i| (self.coordinates[i] - other.coordinates[i]).abs())
            .sum();
        #[cfg(feature = "l_inf_metric")]
        {
            let mut result = (self.coordinates[0] - other.coordinates[0]).abs();
            for i in 1..self.coordinates.len() {
                let tmp_value = (self.coordinates[i] - other.coordinates[i]).abs();
                if tmp_value < result {
                    result = tmp_value;
                }
            }
            return result;
        }
        #[cfg(feature = "l2_metric")]
        return (0..self.coordinates.len())
            .map(|i| (self.coordinates[i] - other.coordinates[i]).powf(2_f32))
            .sum::<f32>()
            .sqrt();
    }

    pub fn parse(point_str: &str) -> Point {
        // (8.0, 9.0)
        let mut result = Point {
            coordinates: Vec::new(),
        };
        let coordinates: Vec<&str> = point_str.trim().split(',').collect();
        result.coordinates.push(
            coordinates[0][1..]
                .parse::<f32>()
                .expect("Error parsing the file!"),
        );
        for i in 1..(coordinates.len() - 1) {
            result.coordinates.push(
                coordinates[i]
                    .trim()
                    .parse::<f32>()
                    .expect("Error parsing the file!"),
            );
        }
        result.coordinates.push(
            coordinates[coordinates.len() - 1][1..(coordinates[coordinates.len() - 1]).len() - 1]
                .parse::<f32>()
                .expect("Error parsing the file!"),
        );
        result
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(").expect("Error writing a point to the output stream!");
        for i in 0..(self.coordinates.len() - 1) {
            write!(f, "{}, ", self.coordinates[i])
                .expect("Error writing a point to the output stream!");
        }
        write!(f, "{})", self.coordinates[self.coordinates.len() - 1])
    }
}
