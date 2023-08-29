use std::cmp::Ordering;
#[cfg(feature = "weighted_vertices")]
use std::collections::{BinaryHeap, HashSet};
#[cfg(feature = "variable_r")]
use std::sync::{Arc, Mutex};

use crate::cell::Cell;

#[derive(Clone)]
pub struct HeapNode {
    #[cfg(not(feature = "variable_r"))]
    pub cells: Vec<*mut Cell>,
    #[cfg(feature = "variable_r")]
    pub cells: Vec<Arc<Cell>>,
    #[cfg(feature = "weighted_vertices")]
    pub solutions_heap: BinaryHeap<InnerHeapNode>,
    #[cfg(feature = "weighted_vertices")]
    pub indices_used: HashSet<Vec<usize>>,
    pub priority: f32,
}

#[derive(Clone)]
#[cfg(feature = "weighted_vertices")]
pub struct InnerHeapNode {
    pub indices: Vec<usize>,
    #[cfg(not(feature = "variable_r"))]
    pub referred_cells: Vec<*mut Cell>,
    #[cfg(feature = "variable_r")]
    pub referred_cells: Vec<Arc<Cell>>,
}

impl HeapNode {
    pub fn new(
        #[cfg(not(feature = "variable_r"))] cells: Vec<*mut Cell>,
        #[cfg(feature = "variable_r")] cells: Vec<Arc<Cell>>,
    ) -> HeapNode {
        HeapNode {
            cells,
            #[cfg(feature = "weighted_vertices")]
            solutions_heap: BinaryHeap::new(),
            #[cfg(feature = "weighted_vertices")]
            indices_used: HashSet::new(),
            priority: 0_f32,
        }
    }
}

#[cfg(feature = "weighted_vertices")]
impl InnerHeapNode {
    pub fn new(
        indices: Vec<usize>,
        #[cfg(not(feature = "variable_r"))] referred_cells: Vec<*mut Cell>,
        #[cfg(feature = "variable_r")] referred_cells: Vec<Arc<Cell>>,
    ) -> InnerHeapNode {
        InnerHeapNode {
            indices,
            referred_cells,
        }
    }
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.total_cmp(&other.priority)
    }
}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.priority.partial_cmp(&other.priority)
    }
}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.cells == other.cells
    }
}

impl Eq for HeapNode {}

#[cfg(feature = "weighted_vertices")]
impl Ord for InnerHeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        unsafe {
            let priority_a: f32 = (0..self.indices.len())
                .map(|i| (*self.referred_cells[i]).points_set[&(i as u32)][self.indices[i]].weight)
                .sum();
            let priority_b: f32 = (0..other.indices.len())
                .map(|i| {
                    (*other.referred_cells[i]).points_set[&(i as u32)][other.indices[i]].weight
                })
                .sum();
            return priority_a.total_cmp(&priority_b);
        }
    }
}

#[cfg(feature = "weighted_vertices")]
impl PartialOrd for InnerHeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        unsafe {
            let priority_a: f32 = (0..self.indices.len())
                .map(|i| (*self.referred_cells[i]).points_set[&(i as u32)][self.indices[i]].weight)
                .sum();
            let priority_b: f32 = (0..other.indices.len())
                .map(|i| {
                    (*other.referred_cells[i]).points_set[&(i as u32)][other.indices[i]].weight
                })
                .sum();
            return priority_a.partial_cmp(&priority_b);
        }
    }
}

#[cfg(feature = "weighted_vertices")]
impl PartialEq for InnerHeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.indices == other.indices
    }
}

#[cfg(feature = "weighted_vertices")]
impl Eq for InnerHeapNode {}
