use rand::Rng;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: &Vec<Layer>) -> Self {
        Self {
            layers: layers.to_vec(),
        }
    }

    pub fn new_random<R: Rng + ?Sized>(layers: &[LayerTopology], rng: &mut R) -> Self {
        assert!(layers.len() > 1);
        let layers = layers
            .windows(2)
            .map(|layers| {
                Layer::new_random(layers[0].neurons, layers[1].neurons, Activation::Tanh, rng)
            })
            .collect();

        Self { layers }
    }

    pub fn to_genes(&self) -> Vec<f32> {
        let mut genes = Vec::new();
        for layer in &self.layers {
            for neuron in &layer.neurons {
                genes.push(neuron.bias);
                genes.extend(&neuron.weights);
            }
        }
        genes
    }

    pub fn from_genes(topology: &[LayerTopology], genes: &[f32]) -> Self {
        assert!(topology.len() > 1);
        let mut index = 0;
        let layers = topology
            .windows(2)
            .map(|layers| {
                let input_size = layers[0].neurons;
                let output_size = layers[1].neurons;
                let neurons = (0..output_size)
                    .map(|_| {
                        let bias = genes[index];
                        index += 1;
                        let weights = genes[index..index + input_size].to_vec();
                        index += input_size;
                        Neuron { bias, weights }
                    })
                    .collect();
                Layer {
                    neurons,
                    activation_function: Activation::Tanh,
                }
            })
            .collect();

        Self { layers }
    }

    pub fn gene_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                layer.neurons.len() * (layer.neurons.first().map_or(0, |n| n.weights.len()) + 1)
            })
            .sum()
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }

    /// Zero-allocation propagation using pre-allocated buffers.
    /// Returns a slice into whichever buffer holds the final output.
    pub fn propagate_into<'a>(
        &self,
        input: &[f32],
        buf_a: &'a mut Vec<f32>,
        buf_b: &'a mut Vec<f32>,
    ) -> &'a [f32] {
        buf_a.clear();
        buf_a.extend_from_slice(input);

        let mut src = buf_a as &mut Vec<f32>;
        let mut dst = buf_b as &mut Vec<f32>;

        for layer in &self.layers {
            dst.clear();
            layer.propagate_into(src, dst);
            std::mem::swap(&mut src, &mut dst);
        }

        // After the loop, src points to the buffer with the final result
        src
    }

    /// Returns the max layer size (useful for pre-allocating buffers)
    pub fn max_layer_size(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.neurons.len())
            .max()
            .unwrap_or(0)
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
    activation_function: Activation,
}

impl Layer {
    pub fn new_random<R: Rng + ?Sized>(
        input_size: usize,
        output_size: usize,
        activation_function: Activation,
        rng: &mut R,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::new_random(input_size, rng))
            .collect();

        Self {
            neurons,
            activation_function,
        }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|x| x.propagate(&inputs, self.activation_function))
            .collect()
    }

    fn propagate_into(&self, inputs: &[f32], out: &mut Vec<f32>) {
        out.extend(
            self.neurons
                .iter()
                .map(|n| n.propagate_slice(inputs, self.activation_function)),
        );
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
        self.propagate_slice(inputs.as_slice(), activation_function)
    }

    fn propagate_slice(&self, inputs: &[f32], activation_function: Activation) -> f32 {
        debug_assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        activation_function.apply(self.bias + output)
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
            Activation::LeakyReLU(a) => {
                if x > 0.0 {
                    x
                } else {
                    a * x
                }
            }
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
        let input_size = self
            .neurons
            .first()
            .map(|neuron| neuron.weights.len())
            .unwrap_or(0);
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

        let layers = vec![
            Layer::new_random(4, 4, Activation::Tanh, &mut rng),
            Layer::new_random(4, 2, Activation::Tanh, &mut rng),
        ];

        let network = Network { layers };

        let inputs: Vec<f32> = vec![0.0, 0.5, 3.2, 1.2];

        network.propagate(inputs);

        println!("Network: {}", network);
    }
}
