use std::{collections::HashMap, hash::Hash, ops::Index};

use macroquad::{
    color::{Color, PINK},
    math::Vec2,
    shapes::{draw_circle, draw_line},
};

use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoadId(pub usize);

#[derive(Clone, Debug)]
pub struct Road {
    pub road_id: RoadId,

    pub points: Vec<Vec2>,
    pub from: Option<NodeId>,
    pub to: Option<NodeId>,
}

#[derive(Clone, Debug)]
pub struct RoadGrid {
    pub roads: Vec<Road>,                         // A collection of the roads
    pub nodes: Vec<Vec2>,                         // Map from NodeId -> Position of that node
    pub next_roads: HashMap<RoadId, Vec<RoadId>>, // Map from RoadId -> Roads that follow
}

impl RoadGrid {
    pub fn new(mut roads: Vec<Road>) -> Self {
        assert!(
            roads.len() > 1,
            "There needs to be more than 1 road in a RoadGrid!"
        );

        let eps = 0.1;
        let mut nodes: Vec<Vec2> = Vec::new();
        // Maps a quantized position to a NodeId
        let mut point_to_node: HashMap<(i32, i32), NodeId> = HashMap::new();

        // Helper closure to quantize point
        let quantize =
            |p: Vec2| -> (i32, i32) { ((p.x / eps).round() as i32, (p.y / eps).round() as i32) };

        // First pass: identify nodes and assign to roads
        for road in roads.iter_mut() {
            let start_pt = *road.get_first_point();
            let end_pt = *road.get_last_point();

            // Process start point
            let start_key = quantize(start_pt);
            let start_node_id = if let Some(&id) = point_to_node.get(&start_key) {
                id
            } else {
                let id = NodeId(nodes.len());
                nodes.push(start_pt);
                point_to_node.insert(start_key, id);
                id
            };
            road.from = Some(start_node_id);

            // Process end point
            let end_key = quantize(end_pt);
            let end_node_id = if let Some(&id) = point_to_node.get(&end_key) {
                id
            } else {
                let id = NodeId(nodes.len());
                nodes.push(end_pt);
                point_to_node.insert(end_key, id);
                id
            };
            road.to = Some(end_node_id);
        }

        // Build helper map: NodeId -> Vec<RoadId> (roads starting at this node)
        let mut out_by_node: HashMap<NodeId, Vec<RoadId>> = HashMap::new();
        for road in &roads {
            if let Some(from_node) = road.from {
                out_by_node.entry(from_node).or_default().push(road.road_id);
            }
        }

        // Second pass: build next_roads
        let mut next_roads: HashMap<RoadId, Vec<RoadId>> = HashMap::new();
        for road in &roads {
            // Insert an entry for every road; dead ends get an empty vec.
            let transitions = road
                .to
                .and_then(|to_node| out_by_node.get(&to_node).cloned())
                .unwrap_or_default();
            next_roads.insert(road.road_id, transitions);
        }

        Self {
            roads,
            nodes,
            next_roads,
        }
    }

    pub fn draw_roads(&self, debug: bool) {
        self.roads.iter().for_each(|x| draw_road(x, debug));
    }
}

impl Index<RoadId> for RoadGrid {
    type Output = Road;
    fn index(&self, index: RoadId) -> &Self::Output {
        self.roads.get(index.0).unwrap_or_else(|| {
            panic!(
                "Could not find road {:?} in Road Grid; max index is {}",
                index,
                self.roads.len().saturating_sub(1)
            )
        })
    }
}

impl Road {
    pub fn new(origin: Vec2, end: Vec2, id: RoadId) -> Self {
        let mut fin_vector: Vec<Vec2> = Vec::new();
        fin_vector.push(origin.clone());

        let mut rng = rand::rng();

        let deviation: f32 = rng.random_range(5..25) as f32;

        let direction_vector = end - origin;

        let steps = 4.0;
        let step_vector = (direction_vector / steps) + Vec2::new(deviation, deviation);

        let p0 = origin;
        let p1 = p0 + step_vector;
        let p2 = p0 + 2.0 * step_vector;
        let p3 = end;

        let mut t = 0.1;
        let (mut a, mut b, mut c, mut d, mut e);

        while t < 1.0 {
            a = p0.lerp(p1, t);
            b = p1.lerp(p2, t);
            c = p2.lerp(p3, t);

            d = a.lerp(b, t);
            e = b.lerp(c, t);

            let fin_point = d.lerp(e, t);

            fin_vector.push(fin_point);
            t += 0.01;
        }
        fin_vector.push(end.clone());

        Road {
            points: fin_vector,
            road_id: id,
            from: None,
            to: None,
        }
    }

    pub fn get_id(&self) -> RoadId {
        self.road_id
    }

    pub fn get_first_point(&self) -> &Vec2 {
        self.points
            .get(0)
            .expect("Attempted to get first point of empty road")
    }

    pub fn get_last_point(&self) -> &Vec2 {
        self.points
            .last()
            .expect("Attempted to get last point of empty road")
    }
}

pub fn draw_road(road: &Road, debug: bool) {
    const THICKNESS: f32 = 60.0;
    const ROAD_COLOR: Color = Color::from_rgba(255, 255, 255, 50);

    let (_start, _end) = (
        road.points
            .first()
            .expect("Road Points not Initialized Correctly"),
        road.points
            .last()
            .expect("Road Points not Initialized Correctly"),
    );

    // Draw Loop
    for i in 0..road.points.len() - 1 {
        let curr = road.points[i];
        let next = road.points[i + 1];

        draw_line(curr.x, curr.y, next.x, next.y, THICKNESS, ROAD_COLOR);

        if debug {
            draw_circle(
                road.points.last().unwrap().x,
                road.points.last().unwrap().y,
                10.0,
                PINK,
            );
        }
    }
}

pub fn generate_road_grid<R: Rng>(roads: usize, rng: &mut R) -> RoadGrid {
    // Roughly split requested roads into a Manhattan-style grid of vertical and horizontal streets.
    let total = roads.max(4) as usize;
    let vertical_count = (total / 2).max(2);
    let horizontal_count = total - vertical_count; // guaranteed >= 2 because total >= 4

    // Space lines across a "city" that is much larger than the current viewport.
    let max_axis = vertical_count.max(horizontal_count) as f32;
    let block_size = 500.0;
    let half_span = block_size * (max_axis - 1.0) * 0.5 + block_size;
    let jitter_fraction = 0.25;

    // Generate evenly spaced positions with a bit of jitter so the grid feels less artificial.
    fn line_positions(
        count: usize,
        half_span: f32,
        jitter_fraction: f32,
        rng: &mut impl Rng,
    ) -> Vec<f32> {
        if count == 0 {
            return Vec::new();
        }
        if count == 1 {
            return vec![0.0];
        }

        let spacing = (half_span * 2.0) / ((count - 1) as f32);
        let jitter_max = spacing * jitter_fraction;

        let mut positions = Vec::with_capacity(count);
        for i in 0..count {
            let base = -half_span + i as f32 * spacing;
            let jitter = if jitter_max > 0.0 {
                rng.random_range((-jitter_max)..jitter_max)
            } else {
                0.0
            };
            positions.push(base + jitter);
        }

        positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        positions
    }

    let city_center = Vec2::new(1920.0 * 0.5, 1080.0 * 0.5);
    let xs = line_positions(vertical_count, half_span, jitter_fraction, rng);
    let ys = line_positions(horizontal_count, half_span, jitter_fraction, rng);

    let mut generated: Vec<Road> = Vec::with_capacity(vertical_count + horizontal_count);
    let mut id: usize = 0;

    // Vertical streets
    for x in xs {
        let start = Vec2::new(city_center.x + x, city_center.y - half_span);
        let end = Vec2::new(city_center.x + x, city_center.y + half_span);
        generated.push(Road::new(start, end, RoadId(id)));
        id += 1;
    }

    // Horizontal streets
    for y in ys {
        let start = Vec2::new(city_center.x - half_span, city_center.y + y);
        let end = Vec2::new(city_center.x + half_span, city_center.y + y);
        generated.push(Road::new(start, end, RoadId(id)));
        id += 1;
    }

    RoadGrid::new(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use macroquad::math::Vec2;

    #[test]
    fn test_road_grid_new() {
        // Create 3 roads:
        // Road 0: (0,0) -> (10,0)
        // Road 1: (10,0) -> (20,0)  (Connects to Road 0 at (10,0))
        // Road 2: (0,0) -> (0,10)   (Connects to Road 0 at (0,0))

        let r0 = Road::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), RoadId(0));
        let r1 = Road::new(Vec2::new(10.0, 0.0), Vec2::new(20.0, 0.0), RoadId(1));
        let r2 = Road::new(Vec2::new(0.0, 0.0), Vec2::new(0.10, 10.0), RoadId(2)); // Use approx 0.0 due to quantization

        let roads = vec![r0, r1, r2];
        let grid = RoadGrid::new(roads);

        // Check node count.
        // Node A at (0,0) - shared by r0 start and r2 start
        // Node B at (10,0) - shared by r0 end and r1 start
        // Node C at (20,0) - r1 end
        // Node D at (0,10) - r2 end
        // Total 4 nodes.
        assert_eq!(grid.nodes.len(), 4, "Should have 4 unique nodes");

        // Check Road 0 connectivity
        // r0 end should connect to r1 start.
        // next_roads[r0] should contain r1.
        let next_r0 = grid.next_roads.get(&RoadId(0));
        assert!(next_r0.is_some(), "Road 0 should have next roads");
        let next_r0 = next_r0.unwrap();
        assert!(next_r0.contains(&RoadId(1)), "Road 0 should lead to Road 1");

        // Road 1 end connects to nothing in this set
        let next_r1 = grid.next_roads.get(&RoadId(1));
        assert!(
            next_r1.is_none() || next_r1.unwrap().is_empty(),
            "Road 1 should not lead anywhere"
        );

        // Road 2 connects to nothing
        let next_r2 = grid.next_roads.get(&RoadId(2));
        assert!(
            next_r2.is_none() || next_r2.unwrap().is_empty(),
            "Road 2 should not lead anywhere"
        );

        // What about roads starting at the same node?
        // r0 starts at (0,0). r2 starts at (0,0).
        // This doesn't affect 'next_roads' unless a road ENDS at (0,0).
        // Let's add a road ending at (0,0).
        // Road 3: (-10, 0) -> (0,0)
    }

    #[test]
    fn test_multiple_outgoing() {
        // Road 0: (-10, 0) -> (0,0)
        // Road 1: (0,0) -> (10,0)
        // Road 2: (0,0) -> (0,10)
        let r0 = Road::new(Vec2::new(-10.0, 0.0), Vec2::new(0.0, 0.0), RoadId(0));
        let r1 = Road::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), RoadId(1));
        let r2 = Road::new(Vec2::new(0.0, 0.0), Vec2::new(0.0, 10.0), RoadId(2));

        let grid = RoadGrid::new(vec![r0, r1, r2]);

        let next_r0 = grid
            .next_roads
            .get(&RoadId(0))
            .expect("Road 0 should have outgoing roads");
        assert!(next_r0.contains(&RoadId(1)));
        assert!(next_r0.contains(&RoadId(2)));
        assert_eq!(next_r0.len(), 2);
    }
}
