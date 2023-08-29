extern crate queues;
use database::Database;
#[cfg(feature = "static")]
use database::DATABASE_FILE;
#[cfg(feature = "variable_r")]
use database::MAX_RECORD_VALUE;
use graph::Graph;
use grid::Grid;
#[cfg(not(feature = "variable_r"))]
use grid::QUERY_RESULT_OUTPUT_FILE;
#[cfg(not(feature = "variable_r"))]
use point::Point;
#[cfg(not(feature = "variable_r"))]
use queues::*;
#[cfg(not(feature = "variable_r"))]
use std::fs::remove_file;
use std::fs::File;
#[cfg(not(feature = "variable_r"))]
use std::io::BufWriter;
#[cfg(not(feature = "variable_r"))]
use std::io::Read;
#[cfg(not(feature = "variable_r"))]
use std::io::Write;
#[cfg(not(feature = "variable_r"))]
use std::path::Path;
#[cfg(feature = "variable_r")]
use std::sync::Arc;
#[cfg(feature = "variable_r")]
use std::sync::Mutex;
#[cfg(feature = "variable_r")]
use std::thread;
use std::time::Instant;

pub mod cell;
pub mod database;
pub mod graph;
pub mod grid;
#[cfg(not(feature = "normal"))]
pub mod heap_node;
pub mod point;

const GRAPH_FILE: &'static str = "input_graph.txt";
#[cfg(not(feature = "variable_r"))]
const OUT_NAME: &'static str = "Test_results.txt";

fn main() {
    let g = Graph::new(&mut File::open(GRAPH_FILE).expect("Error opening the graph file!"));
    #[cfg(not(feature = "variable_r"))]
    let mut output_file;
    #[cfg(not(feature = "variable_r"))]
    if Path::new(OUT_NAME).exists() {
        remove_file(OUT_NAME).expect("Error deleting the output file!");
    }
    #[cfg(not(feature = "variable_r"))]
    {
        output_file =
            BufWriter::new(File::create(OUT_NAME).expect("Error opening the output file!"));
    }
    #[cfg(feature = "static")]
    {
        // Tests with different R
        println!("Running tests with different Rs...");
        test_case(
            &g,
            1.5,
            2,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            1.0,
            2,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            0.5,
            2,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            2.0,
            2,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            4.0,
            2,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );

        // Tests with different epsilon
        println!("Running tests with different epsilons...");
        test_case(
            &g,
            1.5,
            2,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            1.5,
            2,
            0.35,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            1.5,
            2,
            0.065,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            1.5,
            2,
            1.0,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        test_case(
            &g,
            1.5,
            2,
            0.01,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );

        // Tests with different dimensions
        println!("Running tests with different dimensions...");
        remove_file(DATABASE_FILE).expect("Error deleting database file!");
        test_case(
            &g,
            1.5,
            7,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        remove_file(DATABASE_FILE).expect("Error deleting database file!");
        test_case(
            &g,
            1.5,
            3,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        remove_file(DATABASE_FILE).expect("Error deleting database file!");
        test_case(
            &g,
            1.5,
            4,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
        remove_file(DATABASE_FILE).expect("Error deleting database file!");
        test_case(
            &g,
            1.5,
            2,
            0.1,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
    }
    #[cfg(feature = "dynamic")]
    {
        test_case(
            &g,
            1.5,
            2,
            0.01,
            50,
            5,
            #[cfg(not(feature = "normal"))]
            100,
            &mut output_file,
        );
    }
    #[cfg(feature = "variable_r")]
    {
        // Calculate the dataset spread
        let db = Database::new(g.v, 2);
        let spread = db.get_spread();
        // Set all needed parameters
        let eps: f32 = 0.1;
        let r = MAX_RECORD_VALUE * 2_f32.sqrt();
        let end_i = (spread.log(2.0) / (1.0 + eps / 4.0).log(2.0)).trunc() as usize;
        // Create the r vector
        let mut rs: Vec<f32> = Vec::new();
        for i in 0..=end_i {
            rs.push(r / spread * (1.0 + eps / 4.0).powf(i as f32));
        }
        // Creat all the needed grids
        let grids: Arc<Mutex<Vec<Option<Grid>>>> = Arc::new(Mutex::new(Vec::new()));
        for _ in 0..rs.len() {
            grids.lock().expect("Error acquiring the lock!").push(None);
        }
        let max_threads = usize::from(
            thread::available_parallelism().expect("Error getting the number of available cores!"),
        );
        let loop_function = |init: usize,
                             step: usize,
                             grids: Arc<Mutex<Vec<Option<Grid>>>>,
                             rs: &Vec<f32>,
                             eps: f32,
                             g: &Graph,
                             db: &Database| {
            for i in (init..rs.len()).step_by(step) {
                grids.lock().expect("Error acquiring the lock!")[i] =
                    Some(Grid::new(db, g, eps, rs[i]));
            }
        };
        let mut handles: Vec<thread::JoinHandle<()>> = vec![];
        let now = Instant::now();
        for i in 0..max_threads {
            let grids_tmp = Arc::clone(&grids);
            let rs_clone = rs.clone();
            let g_clone = g.clone();
            let db_clone = db.clone();
            handles.push(thread::spawn(move || {
                loop_function(
                    i,
                    max_threads,
                    grids_tmp,
                    &rs_clone,
                    eps,
                    &g_clone,
                    &db_clone,
                );
            }));
        }
        for handle in handles {
            handle.join().expect("Error joining thread!");
        }
        println!(
            "Time to build all grids: {:.5}s",
            now.elapsed().as_secs_f64()
        );

        test_case(
            Arc::clone(&grids),
            &g,
            &rs,
            0.84,
            end_i,
            #[cfg(not(feature = "normal"))]
            100,
        );
        test_case(
            Arc::clone(&grids),
            &g,
            &rs,
            1.27,
            end_i,
            #[cfg(not(feature = "normal"))]
            100,
        );
        test_case(
            Arc::clone(&grids),
            &g,
            &rs,
            1.93,
            end_i,
            #[cfg(not(feature = "normal"))]
            100,
        );
        test_case(
            Arc::clone(&grids),
            &g,
            &rs,
            3.03,
            end_i,
            #[cfg(not(feature = "normal"))]
            100,
        );
    }
}

#[cfg(feature = "static")]
fn test_case(
    g: &Graph,
    r: f32,
    n_dimensions: u8,
    eps: f32,
    #[cfg(not(feature = "normal"))] n_to_report: u32,
    out: &mut BufWriter<File>,
) {
    let err_message = "Error writing to output file!";
    writeln!(out, "Test case R({}), d({}), eps({})", r, n_dimensions, eps).expect(err_message);
    let db = Database::new(g.v, n_dimensions);
    println!("Initializing the grid...");
    let mut now = Instant::now();
    let mut gd = Grid::new(&db, &g, eps, r);
    let mut elapsed_time = now.elapsed();
    writeln!(
        out,
        "Time to build the grid = {:.3}s",
        elapsed_time.as_secs_f64()
    )
    .expect(err_message);
    println!("Answering query...");
    if Path::new(QUERY_RESULT_OUTPUT_FILE).exists() {
        remove_file(QUERY_RESULT_OUTPUT_FILE).expect("Error deleting the output file!");
    }
    now = Instant::now();
    #[cfg(not(feature = "normal"))]
    gd.answer_query(&g, n_to_report);
    #[cfg(feature = "normal")]
    gd.answer_query(&g);
    elapsed_time = now.elapsed();
    writeln!(
        out,
        "Time to answer the query = {:.3}s",
        elapsed_time.as_secs_f64()
    )
    .expect(err_message);
    println!("Checking shapes...");
    let real_approx = check_real_approx(g, r);
    let approximation_ratio: f64;
    if real_approx.0 == 0 {
        approximation_ratio = 0.0;
    } else {
        approximation_ratio = 1.0 + f64::from(real_approx.1) / f64::from(real_approx.0);
    }
    writeln!(
        out,
        "Real shapes = {}\nApprox shapes = {}\nApproximation ratio = {:.5}",
        real_approx.0, real_approx.1, approximation_ratio
    )
    .expect(err_message);
}

#[cfg(feature = "dynamic")]
fn test_case(
    g: &Graph,
    r: f32,
    n_dimensions: u8,
    eps: f32,
    initialization_percentage: u8,
    percentage_to_split: u8,
    #[cfg(not(feature = "normal"))] n_to_report: u32,
    out: &mut BufWriter<File>,
) {
    let err_message = "Error writing to output file!";
    writeln!(
        out,
        "Test case R({}), d({}), eps({}), init_percentage({}), percentage_to_split({})",
        r, n_dimensions, eps, initialization_percentage, percentage_to_split
    )
    .expect(err_message);
    let db = Database::new(g.v, n_dimensions);
    let db_splitted = db.split_points(initialization_percentage);
    println!("Initializing the grid...");
    let mut now = Instant::now();
    let mut gd = Grid::new(&db_splitted.0, &g, eps, r);
    writeln!(
        out,
        "Time to build the grid = {:.3}s",
        now.elapsed().as_secs_f64()
    )
    .expect(err_message);
    let mut to_split = db_splitted.1;
    let mut update_times = Vec::new();
    let mut query_times = Vec::new();
    let mut real_approx_shapes = Vec::new();
    for i in 0..(100 / percentage_to_split) {
        println!("---------------------\nIteration {}", i + 1);
        let split = to_split.split_points(percentage_to_split);
        to_split = split.1;
        let mut n_points_added = 0;
        println!("Adding points...");
        now = Instant::now();
        for pair in split.0.get_data() {
            for p in pair.1 {
                unsafe {
                    Grid::add_point(&mut gd as *mut Grid, *pair.0, p, g);
                }
                n_points_added += 1;
            }
        }
        let mut elapsed_time = now.elapsed().as_secs_f64();
        if n_points_added != 0 {
            elapsed_time /= n_points_added as f64;
        }
        update_times.push(elapsed_time);
        println!("Answering query...");
        if Path::new(QUERY_RESULT_OUTPUT_FILE).exists() {
            remove_file(QUERY_RESULT_OUTPUT_FILE).expect("Error deleting the output file!");
        }
        now = Instant::now();
        #[cfg(feature = "normal")]
        gd.answer_query(g);
        #[cfg(not(feature = "normal"))]
        gd.answer_query(g, n_to_report);
        query_times.push(now.elapsed().as_secs_f64());
        println!("Checking shapes...");
        real_approx_shapes.push(check_real_approx(g, r));
    }
    write!(out, "Update time after each update: ").expect(err_message);
    for update_time in update_times {
        write!(out, "{:.5}s ", update_time).expect(err_message);
    }
    writeln!(out).expect(err_message);
    write!(out, "Query time after each update: ").expect(err_message);
    for query_time in query_times {
        write!(out, "{:.5}s ", query_time).expect(err_message);
    }
    writeln!(out).expect(err_message);
    write!(out, "(Real, approx) shapes after each update: ").expect(err_message);
    for r_a in &real_approx_shapes {
        write!(out, "({}, {}) - ", r_a.0, r_a.1).expect(err_message);
    }
    writeln!(out).expect(err_message);
}

#[cfg(feature = "variable_r")]
fn test_case(
    grids: Arc<Mutex<Vec<Option<Grid>>>>,
    g: &Graph,
    rs: &Vec<f32>,
    radius: f32,
    end_i: usize,
    #[cfg(not(feature = "normal"))] n_to_report: u32,
) {
    let mut l = 0;
    let mut r = end_i;
    while l <= r {
        let m = l + (r - l) / 2;
        if rs[m] < radius {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    println!("Answering query with radius: {}", rs[r]);
    let tmptmp = &mut grids.lock().expect("Error acquiring the lock!");
    let tmp_grid = tmptmp[r].as_mut().unwrap();
    let now = Instant::now();
    #[cfg(feature = "normal")]
    tmp_grid.answer_query(g);
    #[cfg(not(feature = "normal"))]
    tmp_grid.answer_query(g, n_to_report);
    println!(
        "Time to answer the query: {:.5}s",
        now.elapsed().as_secs_f64()
    );
}

#[cfg(not(feature = "variable_r"))]
fn check_real_approx(g: &Graph, r: f32) -> (u32, u32) {
    let mut real_shape: u32 = 0;
    let mut approx_shape: u32 = 0;
    let mut input_file = match File::open(QUERY_RESULT_OUTPUT_FILE) {
        Ok(file) => file,
        Err(_) => {
            return (0, 0);
        }
    };
    let mut shapes_list_string = String::new();
    input_file
        .read_to_string(&mut shapes_list_string)
        .expect("Error reading the database file!");
    let shape_list: Vec<&str> = shapes_list_string.trim().split('\n').collect();

    for shape in shape_list {
        if shape.is_empty() {
            continue;
        }
        let mut shape_vertices = Vec::new();
        let vertices: Vec<&str> = shape.split('|').collect();
        for vertex_str in vertices {
            shape_vertices.push(Point::parse(vertex_str));
        }
        // Queue for BFS
        let mut q: Queue<u32> = queue![];
        // Initialize bool vector for keeping track of visited nodes
        let mut visited = Vec::new();
        for _ in 0..g.v {
            visited.push(false);
        }
        visited[0] = true;
        // Insert vertex into queue
        q.add(0)
            .expect("Error while adding the first element to the queue!");
        // Initialize variables
        let mut real = true;
        // Start BFS loop
        while q.size() > 0 && real {
            let v_j = match q.remove() {
                Ok(num) => num,
                Err(_) => panic!("How is it possible? The queue shouldn't be empty..."),
            };
            for v_h in &g.adj_list[v_j as usize] {
                if !(visited[*v_h as usize]) {
                    visited[*v_h as usize] = true;
                    q.add(*v_h)
                        .expect("Error while adding an element to the queue!");
                }
                if shape_vertices[v_j as usize].distance_from(&(shape_vertices[*v_h as usize])) > r
                {
                    real = false;
                }
            }
        }

        if real {
            real_shape += 1;
        } else {
            approx_shape += 1;
        }
    }
    (real_shape, approx_shape)
}
