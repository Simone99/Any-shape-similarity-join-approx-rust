extern crate queues;
use avl::AvlTreeMap;
#[cfg(not(feature = "normal"))]
use mut_binary_heap::*;
use queues::*;
#[cfg(feature = "weighted_vertices")]
use std::cmp::Reverse;
#[cfg(not(feature = "weighted_vertices"))]
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
#[cfg(all(feature = "weighted_vertices", feature = "variable_r"))]
use std::sync::{Arc, Mutex};
use std::{
    collections::{HashMap, LinkedList},
    io::BufWriter,
};

#[cfg(not(feature = "normal"))]
use crate::heap_node::HeapNode;
#[cfg(feature = "weighted_vertices")]
use crate::heap_node::InnerHeapNode;
use crate::{cell::Cell, database::Database, graph::Graph, point::Point};

pub const QUERY_RESULT_OUTPUT_FILE: &'static str = "query_result.txt";
const POINT_SEPARATOR: &'static str = " | ";
pub struct Grid {
    cells: AvlTreeMap<Vec<i32>, Cell>,
    #[cfg(feature = "normal")]
    active_cells_tree: AvlTreeMap<Vec<i32>, &'static mut Cell>,
    #[cfg(all(not(feature = "normal"), not(feature = "variable_r")))]
    cells_heap: BinaryHeap<Vec<*mut Cell>, HeapNode>,
    #[cfg(all(not(feature = "normal"), feature = "variable_r"))]
    cells_heap: BinaryHeap<Vec<Arc<Cell>>, HeapNode>,
    eps: f32,
    r: f32,
}
#[cfg(feature = "weighted_edges")]
const PRIORITY_FUNCTION: &'static dyn Fn(f32, f32, u32) -> f32 =
    &|distance: f32, r: f32, e: u32| (e as f32) * r / distance;

impl Grid {
    pub fn new(db: &Database, g: &Graph, eps: f32, r: f32) -> Grid {
        let mut result = Grid {
            cells: AvlTreeMap::new(),
            #[cfg(feature = "normal")]
            active_cells_tree: AvlTreeMap::new(),
            #[cfg(not(feature = "normal"))]
            cells_heap: BinaryHeap::new(),
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

    unsafe fn update_recursive(
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
                Self::update_recursive(
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

    unsafe fn update(
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

        Self::update_recursive(
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
                    m: vec![0; g.v as usize],
                    #[cfg(feature = "normal")]
                    m_c: 0,
                    coordinates: cell_coordinates.clone(),
                };
                #[cfg(feature = "cell_distance_vertices")]
                let mut cell_tmp = Cell {
                    points_set: HashMap::new(),
                    m: vec![0; g.v as usize],
                    #[cfg(feature = "normal")]
                    m_c: 0,
                    coordinates: cell_coordinates.clone(),
                    cell_vertices: Vec::new(),
                };
                #[cfg(feature = "cell_distance_center")]
                let mut cell_tmp = Cell {
                    points_set: HashMap::new(),
                    m: vec![0; g.v as usize],
                    #[cfg(feature = "normal")]
                    m_c: 0,
                    coordinates: cell_coordinates.clone(),
                    cell_center: Point {
                        coordinates: Vec::new(),
                        #[cfg(feature = "weighted_vertices")]
                        weight: 0_f32,
                    },
                    eps: 0_f32,
                };

                #[cfg(feature = "cell_distance_vertices")]
                {
                    // If we have to check the cell vertices for the distance
                    let mut core_point = Point {
                        coordinates: Vec::new(),
                        #[cfg(feature = "weighted_vertices")]
                        weight: 0_f32,
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
                        #[cfg(feature = "weighted_vertices")]
                        weight: 0_f32,
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
        #[cfg(not(feature = "weighted_vertices"))]
        // Push without ordering
        (*cell_ptr)
            .points_set
            .get_mut(&color)
            .unwrap()
            .push(p.clone());
        #[cfg(feature = "weighted_vertices")]
        {
            // Sorted insertion
            let tmp_vec = (*cell_ptr).points_set.get_mut(&color).unwrap();
            match tmp_vec.binary_search_by_key(&Reverse(p), |point| Reverse(point)) {
                Ok(_pos) => { /* Element already in the vector */ }
                Err(pos) => tmp_vec.insert(pos, p.clone()),
            }
        }

        (*cell_ptr).m[color as usize] += 1;

        #[cfg(feature = "normal")]
        // Optimization
        if color == 0 && (*cell_ptr).m[color as usize] != 1 {
            (*cell_ptr).m_c += (*cell_ptr).m_c / ((*cell_ptr).m[0] - 1);
            return;
        }

        #[cfg(feature = "normal")]
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

        #[cfg(feature = "weighted_vertices")]
        let lambda_function: &mut dyn FnMut(*mut Grid, &mut Vec<Vec<Option<&mut Cell>>>) =
            &mut |grid: *mut Grid, all_solutions: &mut Vec<Vec<Option<&mut Cell>>>| {
                for sol in all_solutions {
                    #[cfg(not(feature = "variable_r"))]
                    let key: Vec<*mut Cell> = sol
                        .into_iter()
                        .map(|opt| opt.as_mut().unwrap())
                        .map(|cell| *cell as *mut Cell)
                        .collect();
                    #[cfg(feature = "variable_r")]
                    let key: Vec<Arc<Cell>> = sol
                        .into_iter()
                        .map(|opt| opt.as_ref().unwrap())
                        .map(|cell| Arc::new(**cell))
                        .collect();
                    let priority: f32 = (0..g.v)
                        .map(|i| sol[i as usize].as_ref().unwrap().points_set[&i][0].weight)
                        .sum();
                    // Look for the solution inside the heap
                    match (*grid).cells_heap.get_mut(&key) {
                        Some(mut node) => {
                            (*node).priority = priority;
                        }
                        None => {
                            let mut tmp_heapnode = HeapNode::new(key.clone());
                            let indices = vec![0; g.v as usize];
                            tmp_heapnode
                                .solutions_heap
                                .push(InnerHeapNode::new(indices.clone(), key.clone()));
                            tmp_heapnode.indices_used.insert(indices);
                            tmp_heapnode.priority = priority;
                            (*grid).cells_heap.push(key.clone(), tmp_heapnode.clone());
                        }
                    };
                }
            };

        #[cfg(feature = "weighted_edges")]
        let lambda_function: &mut dyn FnMut(*mut Grid, &mut Vec<Vec<Option<&mut Cell>>>) =
            &mut |grid: *mut Grid, all_solutions: &mut Vec<Vec<Option<&mut Cell>>>| {
                for sol in all_solutions {
                    let key: Vec<*mut Cell> = sol
                        .into_iter()
                        .map(|opt| opt.as_mut().unwrap())
                        .map(|cell| *cell as *mut Cell)
                        .collect();
                    let priority = Grid::get_priority(grid, &key.clone(), g, PRIORITY_FUNCTION);
                    // Look for the solution inside the heap
                    match (*grid).cells_heap.get_mut(&key.clone()) {
                        Some(mut node) => {
                            (*node).priority = priority;
                        }
                        None => {
                            (*grid).cells_heap.push(
                                key.clone(),
                                HeapNode {
                                    cells: key,
                                    priority,
                                },
                            );
                        }
                    };
                }
            };

        Self::update(grid, cell, g, color, lambda_function);
    }

    #[cfg(feature = "weighted_edges")]
    fn get_priority(
        grid: *mut Grid,
        sol: &Vec<*mut Cell>,
        g: &Graph,
        priority_function: &dyn Fn(f32, f32, u32) -> f32,
    ) -> f32 {
        unsafe {
            let g_clone = g.clone();
            priority_function(
                g_clone
                    .edge_list
                    .into_iter()
                    .map(|edge| (*sol[edge.0 as usize]).distance_from(&(*(sol[edge.1 as usize]))))
                    .sum(),
                (*grid).r,
                g.e,
            )
        }
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
        #[cfg(not(feature = "weighted_vertices"))]
        let i = match (0..cell.points_set[&color].len()).find_map(|i| {
            if cell.points_set[&color][i] == *p {
                return Some(i);
            }
            None
        }) {
            Some(i) => i,
            None => return,
        };
        #[cfg(feature = "weighted_vertices")]
        let i = match cell.points_set[&color]
            .binary_search_by_key(&Reverse(p), |point| Reverse(point))
        {
            Ok(pos) => pos,
            Err(_pos) => return,
        };

        #[cfg(feature = "normal")]
        if color == 0 && cell.m[0] > 1 {
            cell.m_c -= cell.m_c / cell.m[0];
            cell.m[0] -= 1;
            cell.points_set.get_mut(&0).unwrap().remove(i);
            return;
        }

        let cell_ptr = cell as *mut Cell;

        #[cfg(feature = "normal")]
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

        #[cfg(not(feature = "normal"))]
        let lambda_function: &mut dyn FnMut(*mut Grid, &mut Vec<Vec<Option<&mut Cell>>>) =
            &mut |grid: *mut Grid, all_solutions: &mut Vec<Vec<Option<&mut Cell>>>| {
                // Remove the element
                (*cell_ptr).points_set.get_mut(&color).unwrap().remove(i);
                (*cell_ptr).m[color as usize] -= 1;

                // Update priority
                if i == 0 {
                    // This is the only case i which the cell could be empty, because if we delete a point in another section of the vector
                    // for sure we have other elements stored in the vector, so the cell is not empty.
                    // Check if the cell is now empty
                    let empty = (*cell_ptr).points_set.values().all(|x| x.is_empty());
                    // If it was the first element we need to actually change the priority in the corresponding heap node
                    for sol in all_solutions {
                        // Look for the solution inside the heap
                        #[cfg(not(feature = "variable_r"))]
                        let key: Vec<*mut Cell> = sol
                            .into_iter()
                            .map(|opt| opt.as_mut().unwrap())
                            .map(|cell| *cell as *mut Cell)
                            .collect();
                        #[cfg(feature = "variable_r")]
                        let key: Vec<Arc<Cell>> = sol
                            .into_iter()
                            .map(|opt| opt.as_ref().unwrap())
                            .map(|cell| Arc::new(**cell))
                            .collect();
                        match (*grid).cells_heap.get_mut(&key) {
                            Some(mut node) => {
                                if empty {
                                    (*node).priority = 0.0;
                                } else {
                                    // Check if the deleted point was the last stored in the corresponding relation
                                    #[cfg(feature = "weighted_vertices")]
                                    if (*cell_ptr).points_set[&color].len() == 0 {
                                        (*node).priority = 0.0;
                                    } else {
                                        (*node).priority = (*node).priority - p.weight
                                            + (*cell_ptr).points_set[&color][0].weight;
                                    }
                                    #[cfg(feature = "weighted_edges")]
                                    if (*cell_ptr).points_set[&color].len() == 0 {
                                        (*node).priority = 0.0;
                                    }
                                }
                            }
                            None => {}
                        };
                    }
                }
            };

        Self::update(grid, cell, g, color, lambda_function);
    }

    #[cfg(feature = "normal")]
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
                Self::update(grid_ptr, *solution, g, 0, lambda_function);
            }
        }
    }

    #[cfg(feature = "weighted_vertices")]
    pub fn answer_query(&mut self, g: &Graph, n: u32) {
        let grid_ptr = self as *mut Grid;
        let mut n_to_report = n;
        let err_message = "Error writing the query result file!";
        let mut stream = BufWriter::new(
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(QUERY_RESULT_OUTPUT_FILE)
                .expect("Unable to open file"),
        );

        unsafe {
            while n_to_report > 0 {
                // Report the highest shape
                let mut node = match (*grid_ptr).cells_heap.peek_mut() {
                    Some(node) => node,
                    None => break,
                };
                if (*node).priority == 0.0 {
                    break;
                }
                let sol = (*node).cells.clone();
                let indices = match (*node).solutions_heap.pop() {
                    Some(inner_node) => inner_node.indices,
                    None => break,
                };
                for i in 0..(g.v - 1) {
                    write!(
                        stream,
                        "{}{}",
                        (*sol[i as usize]).points_set[&i][indices[i as usize]],
                        POINT_SEPARATOR
                    )
                    .expect(err_message);
                }
                let i = g.v - 1;
                writeln!(
                    stream,
                    "{}",
                    (*sol[i as usize]).points_set[&i][indices[i as usize]]
                )
                .expect(err_message);

                // Time to update the indices
                let referred_cells = (*node).cells.clone();
                for i in 0..(g.v as usize) {
                    let mut next_combination = Vec::from(indices.clone());
                    next_combination[i] += 1;
                    if next_combination[i] < (*(*node).cells[i]).points_set[&(i as u32)].len()
                        && !(*node).indices_used.contains(&next_combination)
                    {
                        (*node).solutions_heap.push(InnerHeapNode::new(
                            next_combination.clone(),
                            referred_cells.clone(),
                        ));
                        (*node).indices_used.insert(next_combination);
                    }
                }

                // Time to update heapnode priority
                match (*node).solutions_heap.peek() {
                    Some(inner_node) => {
                        (*node).priority = (0..(g.v as usize))
                            .map(|i| {
                                (*(*node).cells[i]).points_set[&(i as u32)][inner_node.indices[i]]
                                    .weight
                            })
                            .sum();
                    }
                    None => {
                        (*node).priority = 0.0;
                    }
                };

                n_to_report -= 1;
            }

            drop(stream);

            // Time to reset all data structures to be ready to answer another query
            for (_, hn) in &mut (*grid_ptr).cells_heap {
                hn.indices_used.clear();
                hn.solutions_heap.clear();
                let reset_value = vec![0; g.v as usize];
                hn.indices_used.insert(reset_value.clone());
                hn.solutions_heap
                    .push(InnerHeapNode::new(reset_value, hn.cells.clone()));
                hn.priority = (0..(g.v as usize))
                    .filter(|i| !(*hn.cells[*i]).points_set[&(*i as u32)].is_empty())
                    .map(|i| (*hn.cells[i]).points_set[&(i as u32)][0].weight)
                    .sum();
            }
        }
    }

    #[cfg(feature = "weighted_edges")]
    pub fn answer_query(&mut self, g: &Graph, n: u32) {
        let grid_ptr = self as *mut Grid;
        let mut n_to_report = n;
        let mut stream = BufWriter::new(
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(QUERY_RESULT_OUTPUT_FILE)
                .expect("Unable to open file"),
        );
        unsafe fn report_all_points(
            g: &Graph,
            solution: &Vec<*mut Cell>,
            pos: usize,
            final_combination: *mut Vec<Option<&Point>>,
            n_to_report: &mut u32,
            output_f: &mut BufWriter<File>,
        ) {
            if *n_to_report <= 0 {
                return;
            }
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
                *n_to_report -= 1;
                return;
            }
            let cell_ptr = solution[pos];
            for p in (*cell_ptr).points_set.get(&(pos as u32)).unwrap_unchecked() {
                (*final_combination)[pos] = Some(p);
                report_all_points(
                    g,
                    solution,
                    pos + 1,
                    final_combination,
                    n_to_report,
                    output_f,
                );
            }
        }
        unsafe {
            while n_to_report > 0 {
                // Report the highest shape
                let mut node = match (*grid_ptr).cells_heap.peek_mut() {
                    Some(node) => node,
                    None => break,
                };
                if (*node).priority == 0.0 {
                    break;
                }
                let sol = (*node).cells.clone();
                let mut point_solution: Vec<Option<&Point>> = vec![None; g.v as usize];
                report_all_points(
                    g,
                    &sol,
                    0,
                    &mut point_solution,
                    &mut n_to_report,
                    &mut stream,
                );

                // Time to update heapnode priority
                (*node).priority = 0.0;
            }
            drop(stream);

            // Time to reset all data structures to be ready to answer another query
            for (_, hn) in &mut (*grid_ptr).cells_heap {
                (*hn).priority = Grid::get_priority(grid_ptr, &hn.cells, g, PRIORITY_FUNCTION);
            }
        }
    }
}
