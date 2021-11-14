use nalgebra::{DefaultAllocator, Matrix, OMatrix, Dynamic, Scalar};
// use rng::Rng;

struct NeuralNet where
    DefaultAllocator: nalgebra::allocator::Allocator<f64, Dynamic, Dynamic> {
    config:   NeuralNetConfig,
    w_hidden: Option<OMatrix<f64, Dynamic, Dynamic>>,
    b_hidden: Option<OMatrix<f64, Dynamic, Dynamic>>,
    w_out:    Option<OMatrix<f64, Dynamic, Dynamic>>,
    b_out:    Option<OMatrix<f64, Dynamic, Dynamic>>,
}

struct NeuralNetConfig {
    input_neurons:  usize,
    output_neurons: usize,
    hidden_neurons: usize,
    num_epochs:     i64,
    learning_rate:  f64,
}

impl NeuralNet {
    fn new(config: NeuralNetConfig) -> NeuralNet {
        NeuralNet {
            config:   config,
            w_hidden: None,
            b_hidden: None,
            w_out:    None,
            b_out:    None,
        }
    }

    fn train(&mut self, x: &OMatrix<f64, Dynamic, Dynamic>, y: &OMatrix<f64, Dynamic, Dynamic>) -> Result<(), ()> {
        

        // let mut rng = thread_rng();
        // let x: u32 = rng.gen();
        let w_hidden = OMatrix::<f64, Dynamic, Dynamic>::new_random(
            self.config.input_neurons,
            self.config.hidden_neurons
        );
        
        let b_hidden = OMatrix::<f64, Dynamic, Dynamic>::new_random(
            1,
            self.config.hidden_neurons
        );
           
        let w_out = OMatrix::<f64, Dynamic, Dynamic>::new_random(
            self.config.hidden_neurons,
            self.config.output_neurons
        );

        let b_out = OMatrix::<f64, Dynamic, Dynamic>::new_random(
            1,
            self.config.output_neurons
        );

        // let output = OMatrix::<f64, Dynamic, Dynamic>::new();

        // let output = OMatrix

        self.w_hidden = Some(w_hidden);
        self.b_hidden = Some(b_hidden);
        self.w_out = Some(w_out);
        self.b_out = Some(b_out);

        Ok(())
    }

    fn backpropagate(
        &mut self,
        x: &OMatrix<f64, Dynamic, Dynamic>,
        y: &OMatrix<f64, Dynamic, Dynamic>
    ) -> Result<(), ()> {

        let mut hiddenLayerInput = OMatrix::<f64, Dynamic, Dynamic>::zeros(
            x.nrows(), 
            self.w_hidden.as_ref().unwrap().ncols(),
        );
        Matrix::mul_to(x, self.w_hidden.as_ref().unwrap(), &mut hiddenLayerInput);
        
        

        Ok(())
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn main() {
    let m = OMatrix::<f64, Dynamic, Dynamic>::new_random(3, 3);
    println!("Hello, world!");
}
