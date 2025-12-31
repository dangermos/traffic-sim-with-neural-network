use rand::Rng;

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

pub struct Network {
    layers: Vec<Layer>
}

impl Network {

    pub fn new_random<R: Rng + ?Sized>(layers: &[LayerTopology], rng: &mut R) -> Self {
        assert!(layers.len() > 1);
        let layers = layers
            .windows(2)
            .map(|layers| Layer::new_random(layers[0].neurons, layers[1].neurons, Activation::ReLU, rng))
            .collect();

        Self { layers }
        
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
    activation_function: Activation,
}

impl Layer {

    fn new_random<R: Rng + ?Sized>(input_size: usize, output_size: usize, activation_function: Activation, rng: &mut R) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::new_random(input_size, rng))
            .collect();
    
        Self { neurons, activation_function }
    }

    fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|x| x.propagate(&inputs, self.activation_function))
            .collect() 
    }

}

pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl Neuron {
    fn propagate(&self, inputs: &Vec<f32>, activation_function: Activation) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        return activation_function.apply(self.bias + output);
    }

    fn new_random<R: Rng + ?Sized>(input_size: usize, rng: &mut R) -> Self {

        let bias = rng.random_range(-1.0..1.0);

        let weights = (0..input_size)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        Self { bias, weights }
    }
}


#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    ReLU,
    LeakyReLU(f32), // slope
    Sigmoid,
    Tanh,
    Softsign,
    Softplus,
}

impl Activation {
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Linear => x,
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(a) => if x > 0.0 { x } else { a * x },
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Softsign => x / (1.0 + x.abs()),
            Activation::Softplus => (1.0 + x.exp()).log(2.0),
        }
    }
}

