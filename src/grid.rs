extern crate queues;
use avl::AvlTreeMap;
use queues::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::{
    collections::{HashMap, LinkedList},
    fs::File,
    io::BufWriter,
};

use crate::{cell::Cell, database::Database, graph::Graph, point::Point};

pub const QUERY_RESULT_OUTPUT_FILE: &'static str = "query_result.txt";
const POINT_SEPARATOR: &'static str = " | ";
pub struct Grid {
    cells: AvlTreeMap<Vec<i32>, Cell>,
    active_cells_tree: AvlTreeMap<Vec<i32>, &'static mut Cell>,
    eps: f32,
    r: f32,
}

impl Grid {
    pub fn new(db: &Database, g: &Graph, eps: f32, r: f32) -> Grid {
        let mut result = Grid {
            cells: AvlTreeMap::new(),
            active_cells_tree: AvlTreeMap::new(),
            eps,
            r,
        };
        let result_ptr = &mut result as *mut Grid;
        for pair in db.get_data().iter() {
            for p in pair.1 {
                unsafe {
                    Self::add_point(result_ptr, *pair.0, p, g);
                }
            }
        }
        result
    }

    unsafe fn update_mc_recursive(
        grid: *mut Grid,
        q: *mut Queue<u32>,
        g: &Graph,
        visited: *mut Vec<bool>,
        cells_by_vertex: *mut Vec<LinkedList<&'static mut Cell>>,
        tree_vec: *mut Vec<&'static mut Cell>,
        solution: *mut Vec<Option<&'static mut Cell>>,
        all_solutions: *mut Vec<Vec<Option<&'static mut Cell>>>,
        color: u32,
    ) {
        if (*q).size() == 0 {
            (*all_solutions).push(Vec::new());
            for el in (*solution).iter_mut() {
                let tmp = el.as_deref_mut().unwrap_unchecked();
                (*all_solutions)
                    .last_mut()
                    .unwrap_unchecked()
                    .push(Some(tmp));
            }
            return;
        }
        let v_j = match (*q).remove() {
            Ok(num) => num,
            Err(_) => panic!(
                "How is it possible? The queue shouldn't be empty after the previous check..."
            ),
        };

        for cell_prime in &mut (*cells_by_vertex)[v_j as usize] {
            let mut erase_neighbor_lists = false;
            let mut non_visited_neighbor = Vec::new();
            let cell_prime_ptr = *cell_prime as *mut Cell;
            (*solution)[v_j as usize] = Some(*cell_prime);
            for v_h in &g.adj_list[v_j as usize] {
                if (*visited)[*v_h as usize] {
                    match &(*solution)[*v_h as usize] {
                        Some(cell) => {
                            if (*cell_prime_ptr).distance_from(cell) > (*grid).r {
                                (*solution)[v_j as usize] = None;
                                erase_neighbor_lists = true;
                                break;
                            }
                        }
                        None => {}
                    }
                } else {
                    non_visited_neighbor.push(v_h);
                    for cell_bar in &mut *tree_vec {
                        let cell_bar_ptr = &mut **cell_bar as *mut Cell;
                        if (*cell_bar_ptr).m[*v_h as usize] > 0
                            && (*cell_prime_ptr).distance_from(&(*cell_bar_ptr)) <= (*grid).r
                        {
                            (*cells_by_vertex)[*v_h as usize].push_back(*cell_bar);
                        }
                    }
                    if (*cells_by_vertex)[*v_h as usize].is_empty() {
                        (*solution)[v_j as usize] = None;
                        erase_neighbor_lists = true;
                        break;
                    }
                }
            }
            if erase_neighbor_lists {
                for v_h in non_visited_neighbor {
                    (*cells_by_vertex)[*v_h as usize].clear();
                }
            } else {
                for v_h in &non_visited_neighbor {
                    (*visited)[**v_h as usize] = true;
                    (*q).add(**v_h)
                        .expect("Error pushing into the queue during BFS!");
                }
                let mut q2 = queue![];
                let mut vec_tmp = Vec::new();
                while (*q).size() > 0 {
                    vec_tmp.push((*q).remove().unwrap_unchecked());
                }
                vec_tmp.reverse();
                for el in vec_tmp {
                    (*q).add(el)
                        .expect("Error pushing into the queue during BFS!");
                    q2.add(el)
                        .expect("Error pushing into the queue during BFS!");
                }
                let q2_ptr = &mut q2 as *mut Queue<u32>;
                Self::update_mc_recursive(
                    grid,
                    q2_ptr,
                    g,
                    visited,
                    cells_by_vertex,
                    tree_vec,
                    solution,
                    all_solutions,
                    color,
                );
                for v_h in non_visited_neighbor {
                    (*visited)[*v_h as usize] = false;
                    (*q).remove().expect("Error while popping from the queue!");
                    (*cells_by_vertex)[*v_h as usize].clear();
                }
                (*solution)[v_j as usize] = None;
            }
        }
    }

    unsafe fn update_mc(
        grid: *mut Grid,
        cell: &'static mut Cell,
        g: &Graph,
        color: u32,
        lambda: &mut dyn FnMut(*mut Grid, &mut Vec<Vec<Option<&mut Cell>>>),
    ) {
        // Queue for BFS
        let mut q: Queue<u32> = queue![];
        // Initialize bool vector for keeping track of visited nodes
        let mut visited = Vec::new();
        for _ in 0..g.v {
            visited.push(false);
        }
        visited[color as usize] = true;
        // Lists to keep track of the cells assigned to all vertices
        let mut cells_by_vertex: Vec<LinkedList<&mut Cell>> = Vec::new();
        for _ in 0..g.v {
            cells_by_vertex.push(LinkedList::new());
        }
        cells_by_vertex[color as usize].push_back(cell);
        // Insert vertex into queue
        q.add(color)
            .expect("Error while adding the first element to the queue!");
        // Take all available cells
        let mut tree_vec: Vec<&mut Cell> = (*grid).cells.values_mut().collect();

        let mut solution: Vec<Option<&mut Cell>> = Vec::new();
        for _ in 0..g.v {
            solution.push(None);
        }

        let mut all_solutions: Vec<Vec<Option<&mut Cell>>> = Vec::new();

        Self::update_mc_recursive(
            grid,
            &mut q,
            g,
            &mut visited,
            &mut cells_by_vertex,
            &mut tree_vec,
            &mut solution,
            &mut all_solutions,
            color,
        );

        if !all_solutions.is_empty() {
            lambda(grid, &mut all_solutions);
        }
    }

    pub unsafe fn add_point(grid: *mut Grid, color: u32, p: &Point, g: &Graph) {
        // Create a temporary cell with corresponding coordinates
        let mut cell_coordinates: Vec<i32> = Vec::new();
        for coordinate in &p.coordinates {
            cell_coordinates.push((coordinate / (*grid).eps).trunc() as i32);
        }

        // Check if the cell already exists
        let cell = match (*grid).cells.get_mut(&cell_coordinates) {
            Some(cell) => cell,
            None => {
                #[cfg(all(
                    not(feature = "cell_distance_vertices"),
                    not(feature = "cell_distance_center")
                ))]
                let mut cell_tmp = Cell {
                    points_set: HashMap::new(),
                    m: Vec::new(),
                    m_c: 0,
                    coordinates: cell_coordinates.clone(),
                };
                #[cfg(feature = "cell_distance_vertices")]
                let mut cell_tmp = Cell {
                    points_set: HashMap::new(),
                    m: Vec::new(),
                    m_c: 0,
                    coordinates: cell_coordinates.clone(),
                    cell_vertices: Vec::new(),
                };
                #[cfg(feature = "cell_distance_center")]
                let mut cell_tmp = Cell {
                    points_set: HashMap::new(),
                    m: Vec::new(),
                    m_c: 0,
                    coordinates: cell_coordinates.clone(),
                    cell_center: Point {
                        coordinates: Vec::new(),
                    },
                    eps: 0_f32,
                };
                for _ in 0..g.v {
                    cell_tmp.m.push(0);
                }

                #[cfg(feature = "cell_distance_vertices")]
                {
                    // If we have to check the cell vertices for the distance
                    let mut core_point = Point {
                        coordinates: Vec::new(),
                    };
                    for coordinate_i in &cell_coordinates {
                        core_point
                            .coordinates
                            .push(*coordinate_i as f32 * (*grid).eps);
                    }

                    unsafe fn find_vertices(
                        cell_vertices: *mut Vec<Point>,
                        core_point: *mut Point,
                        i: usize,
                        k: usize,
                        grid: *mut Grid,
                    ) {
                        if i >= k {
                            (*cell_vertices).push((*core_point).clone());
                            return;
                        }
                        (*core_point).coordinates[i] += (*grid).eps;
                        find_vertices(cell_vertices, core_point, i + 1, k, grid);
                        (*core_point).coordinates[i] -= (*grid).eps;
                        find_vertices(cell_vertices, core_point, i + 1, k, grid);
                    }
                    find_vertices(
                        &mut (cell_tmp.cell_vertices) as *mut Vec<Point>,
                        &mut core_point as *mut Point,
                        0,
                        cell_tmp.coordinates.len(),
                        grid,
                    );
                }

                #[cfg(feature = "cell_distance_center")]
                {
                    // If we have to check the cell vertices for the distance
                    let mut core_point = Point {
                        coordinates: Vec::new(),
                    };
                    for coordinate_i in &cell_coordinates {
                        core_point
                            .coordinates
                            .push(*coordinate_i as f32 * (*grid).eps);
                    }

                    let incr: f32 = (*grid).eps / 2_f32;
                    for dim in &core_point.coordinates {
                        cell_tmp.cell_center.coordinates.push(dim + incr);
                    }
                    cell_tmp.eps = (*grid).eps;
                }

                (*grid).cells.insert(cell_coordinates.clone(), cell_tmp);
                (*grid).cells.get_mut(&cell_coordinates).unwrap()
            }
        };

        let cell_ptr = &mut *cell as *mut Cell;

        if !(*cell_ptr).points_set.contains_key(&color) {
            (*cell_ptr).points_set.insert(color, Vec::new());
        }
        (*cell_ptr)
            .points_set
            .get_mut(&color)
            .unwrap()
            .push(p.clone());
        (*cell_ptr).m[color as usize] += 1;

        // Optimization

        if color == 0 && (*cell_ptr).m[color as usize] != 1 {
            (*cell_ptr).m_c += (*cell_ptr).m_c / ((*cell_ptr).m[0] - 1);
            return;
        }

        let lambda_function: &mut dyn FnMut(*mut Grid, &mut Vec<Vec<Option<&mut Cell>>>) =
            &mut |grid: *mut Grid, all_solutions: &mut Vec<Vec<Option<&mut Cell>>>| {
                for sol in all_solutions {
                    let mut n_solutions: u64 = 1;
                    for i in 1..g.v {
                        n_solutions *= sol[i as usize].as_ref().unwrap().m[i as usize];
                    }
                    let sol_ptr = sol as *mut Vec<Option<&mut Cell>>;
                    unsafe {
                        let tmp = (*sol_ptr)[0].as_mut().unwrap_unchecked();
                        let cell_ptr = *tmp as *mut Cell;
                        (*cell_ptr).m_c += (*cell_ptr).m[0] * n_solutions;
                        (*grid)
                            .active_cells_tree
                            .insert((*cell_ptr).coordinates.clone(), &mut (*cell_ptr));
                    }
                }
            };

        Self::update_mc(grid, cell, g, color, lambda_function);
    }

    pub unsafe fn delete_point(grid: *mut Grid, color: u32, p: &Point, g: &Graph) {
        // Create a temporary cell with corresponding coordinates
        let mut cell_coordinates: Vec<i32> = Vec::new();
        for coordinate in &p.coordinates {
            cell_coordinates.push((coordinate / (*grid).eps).trunc() as i32);
        }

        // Check if the cell already exists
        let cell = match (*grid).cells.get_mut(&cell_coordinates) {
            Some(cell) => cell,
            None => return,
        };
        // Look for the element to delete
        let i = match (0..cell.points_set[&color].len()).find_map(|i| {
            if cell.points_set[&color][i] == *p {
                return Some(i);
            }
            None
        }) {
            Some(i) => i,
            None => return,
        };

        if color == 0 && cell.m[0] > 1 {
            cell.m_c -= cell.m_c / cell.m[0];
            cell.m[0] -= 1;
            cell.points_set.get_mut(&0).unwrap().remove(i);
            return;
        }

        let cell_ptr = cell as *mut Cell;

        //Update mc
        let lambda_function: &mut dyn FnMut(*mut Grid, &mut Vec<Vec<Option<&mut Cell>>>) =
            &mut |grid: *mut Grid, all_solutions: &mut Vec<Vec<Option<&mut Cell>>>| {
                // Remove the element

                (*cell_ptr).points_set.get_mut(&color).unwrap().remove(i);
                (*cell_ptr).m[color as usize] -= 1;

                // Update m_c for all cells in cells_by_vertex[0]
                for sol in all_solutions {
                    let mut n_solutions: u64 = 1;
                    for i in 1..g.v {
                        n_solutions *= sol[i as usize].as_ref().unwrap().m[i as usize];
                    }
                    let sol_ptr = sol as *mut Vec<Option<&mut Cell>>;
                    unsafe {
                        let tmp = (*sol_ptr)[0].as_mut().unwrap_unchecked();
                        let cell_ptr = *tmp as *mut Cell;
                        (*cell_ptr).m_c += (*cell_ptr).m[0] * n_solutions;
                        if (*cell_ptr).m_c == 0 {
                            (*grid).active_cells_tree.remove(&(*cell_ptr).coordinates);
                        }
                    }
                }
                // Handle the case in which the cell is empty
                // Check if the cell is now empty
                if (*cell_ptr).points_set.values().all(|x| x.is_empty()) {
                    (*grid).cells.remove(&(*cell_ptr).coordinates);
                }
            };

        Self::update_mc(grid, cell, g, color, lambda_function);
    }

    pub fn answer_query(&mut self, g: &Graph) {
        let grid_ptr = self as *mut Grid;
        unsafe {
            let lambda_function: &mut dyn FnMut(*mut Grid, &mut Vec<Vec<Option<&mut Cell>>>) =
                &mut |_grid: *mut Grid, all_solutions: &mut Vec<Vec<Option<&mut Cell>>>| {
                    unsafe fn report_all_points(
                        g: &Graph,
                        solution: &mut Vec<Option<&mut Cell>>,
                        pos: usize,
                        final_combination: *mut Vec<Option<&Point>>,
                        output_f: &mut BufWriter<File>,
                    ) {
                        if pos >= g.v as usize {
                            for i in 0..(g.v - 1) {
                                write!(
                                    output_f,
                                    "{}{}",
                                    (*final_combination)[i as usize].unwrap_unchecked(),
                                    POINT_SEPARATOR
                                )
                                .expect("Error writing the query result file!");
                            }
                            writeln!(
                                output_f,
                                "{}",
                                (*final_combination)[(g.v - 1) as usize].unwrap_unchecked()
                            )
                            .expect("Error writing the query result file!");
                            return;
                        }
                        let cell_ptr = *(solution[pos].as_mut().unwrap_unchecked()) as *mut Cell;
                        for p in (*cell_ptr).points_set.get(&(pos as u32)).unwrap_unchecked() {
                            (*final_combination)[pos] = Some(p);
                            report_all_points(g, solution, pos + 1, final_combination, output_f);
                        }
                    }

                    let mut stream = BufWriter::new(
                        OpenOptions::new()
                            .append(true)
                            .create(true)
                            .open(QUERY_RESULT_OUTPUT_FILE)
                            .expect("Unable to open file"),
                    );
                    for sol in all_solutions {
                        let mut point_solution: Vec<Option<&Point>> = Vec::new();
                        for _ in 0..g.v {
                            point_solution.push(None);
                        }
                        report_all_points(g, sol, 0, &mut point_solution, &mut stream);
                    }
                };
            for solution in (*grid_ptr).active_cells_tree.values_mut() {
                Self::update_mc(grid_ptr, *solution, g, 0, lambda_function);
            }
        }
    }
}
