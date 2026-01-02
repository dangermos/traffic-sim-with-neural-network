use rand::Rng;
use std::fmt;

#[derive(Debug)]
pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>
}

impl Network {

    pub fn new(layers: &Vec<Layer>) -> Self {
        Self {
            layers: layers.to_vec()
        }
    }

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

impl Clone for Network {
    fn clone(&self) -> Self {
        Self { layers: self.layers.clone() }
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
    activation_function: Activation,
}

impl Layer {

    pub fn new_random<R: Rng + ?Sized>(input_size: usize, output_size: usize, activation_function: Activation, rng: &mut R) -> Self {
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
#[derive(Debug, Clone)]
/// Neuron of a Neural Network.
/// Bias is a single f32 while weights is a dynamic Vec of f32's.
/// This is because a neuron gets a weight per input to the neuron. 
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

const FLOAT_PRECISION: usize = 4;

fn write_f32(f: &mut fmt::Formatter<'_>, value: f32) -> fmt::Result {
    write!(f, "{:.*}", FLOAT_PRECISION, value)
}

impl fmt::Display for LayerTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LayerTopology(neurons={})", self.neurons)
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Network(layers={})", self.layers.len())?;
        for (index, layer) in self.layers.iter().enumerate() {
            write!(f, "\n  [{}] ", index)?;
            let layer_string = layer.to_string();
            let mut lines = layer_string.lines();
            if let Some(first_line) = lines.next() {
                write!(f, "{}", first_line)?;
            }
            for line in lines {
                write!(f, "\n  {}", line)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let input_size = self.neurons.first().map(|neuron| neuron.weights.len()).unwrap_or(0);
        write!(
            f,
            "Layer(inputs={}, outputs={}, activation={})",
            input_size,
            self.neurons.len(),
            self.activation_function
        )?;
        for (index, neuron) in self.neurons.iter().enumerate() {
            write!(f, "\n  [{}] {}", index, neuron)?;
        }
        Ok(())
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Neuron(bias=")?;
        write_f32(f, self.bias)?;
        write!(f, ", weights=[")?;
        for (index, weight) in self.weights.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write_f32(f, *weight)?;
        }
        write!(f, "])")
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::Linear => write!(f, "Linear"),
            Activation::ReLU => write!(f, "ReLU"),
            Activation::LeakyReLU(slope) => {
                write!(f, "LeakyReLU(")?;
                write_f32(f, *slope)?;
                write!(f, ")")
            }
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::Softsign => write!(f, "Softsign"),
            Activation::Softplus => write!(f, "Softplus"),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn test_new_network() {

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let layers = 
        vec![
            Layer::new_random(4, 4, Activation::Tanh, &mut rng),
            Layer::new_random(4, 2, Activation::Tanh, &mut rng)
        ];

        let network = Network {layers};


        let inputs: Vec<f32> = vec![0.0, 0.5, 3.2, 1.2];

        network.propagate(inputs);

        println!("Network: {}", network);

        
    }
}
