use dep::std;
global scaling_factor = 16;  
global neurons_per_layer_0 = [60, 40, 2];  
global num_0 = 2582;  
global fnn_out_len_0 = 2;  

fn main(mut inputs_0 : pub [u16; num_0]) {

    let mut fnn_out_0 : [u16; fnn_out_len_0] = [0; fnn_out_len_0];
    fnn_out_0 = forward(inputs_0, neurons_per_layer_0, fnn_out_len_0, fnn_out_0);
    std::println(fnn_out_0);


}

fn forward<N, M, D> (mut inputs : [u16; N], neurons_per_layer : [comptime u16; M], fnn_out_len: comptime u16, mut fnn_out : [u16; D]) -> [u16; D] {
    // inputs: [neuron0-0, neuron0-1, ..., w1-0-0, w1-0-1, ..., w1-n-0, w1-n-1, ..., b1-0, b1-1, ...,neuron1-0, neuron1-1, ..., w2-0-0, w2-0-1, ..., w2-n-0, w2-n-1, ..., b2-0, b2-1, ...]
    let mut index_layer = 0;
    for layer in 1 .. neurons_per_layer.len() {
    	index_layer += neurons_per_layer[layer - 1] * (neurons_per_layer[layer]+1);
        for i in 0 .. neurons_per_layer[layer] {
            let mut val = 0; 
            let mut index_b = index_layer + i;
            let mut index_j = index_layer - neurons_per_layer[layer - 1] * (neurons_per_layer[layer]+1);
            for j in 0 .. neurons_per_layer[layer-1] {
                let mut index_neuron = index_j+j;
                let mut index_w = index_neuron+neurons_per_layer[layer-1]*(i+1);
                val += (inputs[index_neuron]*inputs[index_w]/scaling_factor);
            }
            val += inputs[index_b];
            inputs[index_b] = activation(val);
        }
    }
    for i in 0 .. fnn_out_len {
        fnn_out[i] = inputs[inputs.len() as u16 - fnn_out_len + i];
    }
    fnn_out
}
 
fn activation(x: u16) -> u16 {
    let mut result = 0;
    if x>result {
        result = x;
    }
    result
}

#[test]
fn test_main() {
    let inputs_0:[u16; num_0]=[0; num_0];
    main(inputs_0);
}