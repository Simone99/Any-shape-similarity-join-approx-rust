use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

use crate::point::Point;

pub const DATABASE_FILE: &'static str = "input_database.txt";
const SEPARATOR: &'static str = "**************************************************************";
const N_MAX_RECORDS: i32 = 100;
pub const MAX_RECORD_VALUE: f32 = 10.0;

fn generate_random_point(d: u8, die: &Uniform<f32>, rng: &mut ThreadRng) -> Point {
    let mut result = Point {
        coordinates: Vec::new(),
    };
    for _ in 0..d {
        result.coordinates.push(die.sample(rng));
    }
    result
}

#[derive(Clone)]
pub struct Database {
    data: HashMap<u32, Vec<Point>>,
}

impl Database {
    pub fn new(n_sets: u32, d: u8) -> Database {
        let mut result = Database {
            data: HashMap::new(),
        };
        if !Path::new(DATABASE_FILE).exists() {
            let err_message = "Error writing to output file!";
            let mut rng = rand::thread_rng();
            let die = Uniform::from(0_f32..MAX_RECORD_VALUE);
            let mut database_file =
                File::create(DATABASE_FILE).expect("Error creating the database file!");
            for i in 0..n_sets {
                writeln!(database_file, "{}", i).expect(err_message);
                let mut tmp = Vec::new();
                for _ in 0..N_MAX_RECORDS {
                    let tmp_point = generate_random_point(d, &die, &mut rng);
                    writeln!(database_file, "{}", tmp_point).expect(err_message);
                    tmp.push(tmp_point);
                }
                writeln!(database_file, "{}", SEPARATOR).expect(err_message);
                result.data.insert(i, tmp);
            }
        } else {
            let mut database_file =
                File::open(DATABASE_FILE).expect("Error opening the database file!");
            let mut database_string = String::new();
            database_file
                .read_to_string(&mut database_string)
                .expect("Error reading the database file!");
            let mut database_relations: Vec<&str> =
                database_string.trim().split(SEPARATOR).collect();

            database_relations.remove(database_relations.len() - 1);

            for relation in database_relations {
                let records: Vec<&str> = relation.trim().split('\n').collect();
                let color = records[0].parse::<u32>().expect("Error parsing the file!");
                let mut values = Vec::new();
                for i in 1..records.len() {
                    values.push(Point::parse(records[i]));
                }
                result.data.insert(color, values);
            }
        }
        result
    }

    pub fn get_data(&self) -> &HashMap<u32, Vec<Point>> {
        &self.data
    }

    #[cfg(feature = "dynamic")]
    pub fn split_points(&self, percentage: u8) -> (Database, Database) {
        let mut db1 = Database {
            data: HashMap::new(),
        };
        let mut db2 = Database {
            data: HashMap::new(),
        };
        let mut rng = rand::thread_rng();
        let die = Uniform::from(0_f32..100_f32);
        for pair in &self.data {
            let mut values1 = Vec::new();
            let mut values2 = Vec::new();
            for p in pair.1 {
                if die.sample(&mut rng) < percentage as f32 {
                    values1.push(p.clone());
                } else {
                    values2.push(p.clone());
                }
            }
            db1.data.insert(*pair.0, values1);
            db2.data.insert(*pair.0, values2);
        }
        (db1, db2)
    }

    #[cfg(feature = "variable_r")]
    pub fn get_spread(&self) -> f32 {
        let mut max_distance = 0_f32;
        let mut min_distance = f32::MAX;
        let tmp_points: Vec<&Point> = (&self.data).into_iter().flat_map(|entry| entry.1).collect();
        for i in 0..(tmp_points.len() - 1) {
            for j in (i + 1)..tmp_points.len() {
                let distance = tmp_points[i].distance_from(&tmp_points[j]);
                if distance > max_distance {
                    max_distance = distance;
                } else if distance < min_distance {
                    min_distance = distance;
                }
            }
        }
        max_distance / min_distance
    }
}
