use std::ops::Index;

use macroquad::{color::{BLACK, Color, PINK, RED}, math::Vec2, shapes::{draw_circle, draw_line}, window::{screen_height, screen_width}};
use rand::{Rng, rng};

use crate::cars::Destination;


#[derive(Clone, Debug)]
pub struct Road {
    points: Vec<Vec2>,
    id: u16,
}

#[derive(Clone, Debug)]
pub struct RoadGrid {
    pub roads: Vec<Road>,
    destinations: Vec<Destination>
}



impl RoadGrid {
    pub fn new(roads: Vec<Road>) -> Self {

        let destinations = roads.iter().map(
            |x| Destination {position: *x.get_first_point()}
        ).collect::<Vec<Destination>>();

        Self {
            destinations,
            roads 
        }
    }

    pub fn draw_roads(&self, debug: bool) {
        self.roads.iter().for_each(
            |x| draw_road(x, debug)
        );
    }

    pub fn get_destinations(&self) -> &Vec<Destination> {
        &self.destinations
    }
}

impl Index<u16> for RoadGrid {
    type Output = Road;
    fn index(&self, index: u16) -> &Self::Output {
        self.roads.iter().find(
            |x| x.get_id() == index
        ).unwrap_or(&self.roads[0])
    }
}

impl Road {
    pub fn new(origin: Vec2, end: Vec2, id: u16) -> Self {

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
    
    const THICKNESS: f32 = 60.0; 
    const ROAD_COLOR: Color = Color::from_rgba(255, 255, 255, 50);

    let (_start, _end) = (road.points.first().expect("Road Points not Initialized Correctly"),
                                        road.points.last().expect("Road Points not Initialized Correctly"));
                        


    // Draw Loop
    
    for i in 0..road.points.len() - 1 {
        let curr = road.points[i];
        let next = road.points[i + 1];

        draw_line(curr.x, curr.y, next.x, next.y, THICKNESS, ROAD_COLOR);

         
        if debug {
            draw_circle(road.points.last().unwrap().x, road.points.last().unwrap().y, 10.0, PINK);
            /* 
            if (i + 1) % 2 == 0 {
                draw_line(curr.x, curr.y, next.x, next.y, THICKNESS / 4.0, BLACK);
            }
            */
        }


    }

}

pub fn generate_road_grid(roads: i32) -> RoadGrid {

    let max_x = screen_width() as i32;
    let max_y = screen_height() as i32;

    let mut rng = rng();


    let mut r: Vec<Road> = vec![];

    // Generates (roads) amount of random roads
    let mut i: i32 = 0;

    while i < roads {

        let rand_x1 = rng.random_range(0..max_x) as f32;
        let rand_y1 = rng.random_range(0..max_y) as f32;
        let rand_x2 = rng.random_range(0..max_x) as f32;
        let rand_y2 = rng.random_range(0..max_y) as f32;


        let origin = Vec2::new(rand_x1, rand_y1);
        let end = Vec2::new(rand_x2, rand_y2);
    
        r.push(Road::new(
            origin, end,
            i as u16));
        i+= 1;
    }

    fn close(road1: &Road, road2: &Road) -> bool {

        let eps = 5.0;

        if ((road1.points.first().unwrap().x - road2.points.first().unwrap().x).abs() < eps) || 
           ((road1.points.first().unwrap().y - road2.points.first().unwrap().y).abs() < eps) 
           {
            true
        }
        else {
            false
        }


    }

    let mut filtered: Vec<Road> = Vec::with_capacity(r.len());
    for road in r.drain(..) {
        if filtered.iter().any(|existing| close(existing, &road)) {
            continue;
        }
        filtered.push(road);
    }
    r = filtered;
    
    // Now lets connect them!
    let mut temp = vec![];

    for x in r.windows(2) {
        temp.push(Road::new(*x[0].points.last().unwrap(), *x[1].get_first_point(), i as u16));
        i += 1;
    }

    r.append(&mut temp);

    RoadGrid::new(r)



}
