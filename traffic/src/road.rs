use macroquad::{color::{BLACK, Color, PINK, RED}, math::Vec2, shapes::{draw_circle, draw_line}, window::{screen_height, screen_width}};
use rand::{Rng, rng};


#[derive(Clone, Debug)]
pub struct Road {
    points: Vec<Vec2>,
    id: u16,
}

#[derive(Clone, Debug)]
pub struct RoadGrid {
    pub roads: Vec<Road>
}

impl RoadGrid {
    pub fn new(roads: Vec<Road>) -> Self {
        Self {
            roads 
        }
    }

    pub fn draw_roads(&self, debug: bool) {
        self.roads.iter().for_each(
            |x| draw_road(x, debug)
        );
    }
}


impl Road {
    pub fn new(origin: Vec2, end: Vec2, id: u16) -> Self {

        let mut fin_vector: Vec<Vec2> = Vec::new();
        fin_vector.push(origin.clone());

        let mut rng = rand::rng();

        let deviation: f32 = rng.random_range(5..50) as f32;

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

        println!("Made Road with {} points", fin_vector.len());

        Road { points: fin_vector, id}

    }

    pub fn get_id(&self) -> u16 {
        self.id
    }

    pub fn get_first_point(&self) -> &Vec2 {
        self.points.get(0).expect("Attempted to get point of empty road")
    }
 
    
}

pub fn draw_road(road: &Road, debug: bool) {
    
    const THICKNESS: f32 = 5.0; 
    const ROAD_COLOR: Color = Color::from_rgba(255, 255, 255, 50);

    let (_start, _end) = (road.points.first().expect("Road Points not Initialized Correctly"),
                                        road.points.last().expect("Road Points not Initialized Correctly"));
                        


    // Draw Loop
    
    for i in 0..road.points.len() - 1 {
        let curr = road.points[i];
        let next = road.points[i + 1];
        if debug {draw_circle(curr.x, curr.y, 1.0, RED);}

        draw_line(curr.x, curr.y, next.x, next.y, THICKNESS, ROAD_COLOR);

        if (i + 1) % 2 == 0 {
            draw_line(curr.x, curr.y, next.x, next.y, THICKNESS / 4.0, BLACK);
        }


    }


    if debug {draw_circle(road.points.last().unwrap().x, road.points.last().unwrap().y, 1.0, PINK);}

}

pub fn generate_road_grid(roads: i32) -> RoadGrid {

    let max_x = screen_width() as i32;
    let max_y = screen_height() as i32;

    let mut rng = rng();


    let mut r: Vec<Road> = vec![];

    for i in 0..roads {
        let rand_x1 = rng.random_range(0..max_x) as f32;
        let rand_y1 = rng.random_range(0..max_y) as f32;
        let rand_x2 = rng.random_range(0..max_x) as f32;
        let rand_y2 = rng.random_range(0..max_y) as f32;
    
        r.push(Road::new(Vec2::new(rand_x1, rand_y1), Vec2::new(rand_x2, rand_y2), i as u16));
    }


    RoadGrid { roads: r }



}