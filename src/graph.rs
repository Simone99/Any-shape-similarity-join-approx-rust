use std::io::Read;
use std::{collections::LinkedList, fs::File};

#[derive(Clone)]
pub struct Graph {
    pub v: u32,
    pub e: u32,
    pub adj_list: Vec<LinkedList<u32>>,
}

impl Graph {
    pub fn new(input_file: &mut File) -> Graph {
        let mut result = Graph {
            v: 0,
            e: 0,
            adj_list: Vec::new(),
        };
        let mut graph_string = String::new();
        input_file
            .read_to_string(&mut graph_string)
            .expect("Error reading the database file!");
        let graph_data: Vec<&str> = graph_string.trim().split('\n').collect();

        //database_relations.remove(database_relations.len() - 1);

        result.v = graph_data[0]
            .parse::<u32>()
            .expect("Error parsing the file!");
        result.e = graph_data[1]
            .parse::<u32>()
            .expect("Error parsing the file!");
        for _ in 0..result.v {
            result.adj_list.push(LinkedList::new());
        }
        for i in 2..graph_data.len() {
            let str_edge: Vec<&str> = graph_data[i].split(' ').collect();
            let u = str_edge[0].parse::<u32>().expect("Error parsing the file!");
            let v = str_edge[1].parse::<u32>().expect("Error parsing the file!");
            result.adj_list[u as usize].push_back(v);
            result.adj_list[v as usize].push_back(u);
        }
        result
    }
}
