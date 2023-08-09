import random,json,os

def fnn_input_count(network):
    num = network[0]
    for layer in range(1, len(network)):
        num += network[layer]*(network[layer-1]+1)
    print(num)
    return num

def fnn_inputs_value(network):
    inputs = []
    str_inputs = []
    for i in range(fnn_input_count(network)):
        # v = random.randint(1,10)
        v = 0
        inputs.append(v)
        str_inputs.append(str(v))
    # print(inputs)
    str_inputs = str(json.dumps(str_inputs))
    # print(str(json.dumps(str_inputs)))
    # with open("../config/record.json","w") as f:
    #     json.dump(str_inputs,f)
    return inputs, str_inputs

def create_noir_code(network, path=''):
    num = fnn_input_count(network)
    inputs, str_inputs = fnn_inputs_value(network)
    noir_code = 'use dep::std;\nglobal neurons_per_layer = '+str(network)+';\nglobal num = '+str(num)+';\nglobal scaling_factor = 16;\n'
    noir_code+='fn main(mut inputs : pub [u16; num]) {\n\
    let mut index_layer = 0;\n'
    noir_code+='    for layer in 1 .. neurons_per_layer.len() {\n\
    	index_layer += neurons_per_layer[layer - 1] * (neurons_per_layer[layer]+1);\n\
        for i in 0 .. neurons_per_layer[layer] {\n\
            let mut val = 0; \n\
            let mut index_b = index_layer + i;\n\
            let mut index_j = index_layer - neurons_per_layer[layer - 1] * (neurons_per_layer[layer]+1);\n\
            for j in 0 .. neurons_per_layer[layer-1] {\n\
                let mut index_neuron = index_j+j;\n\
                let mut index_w = index_neuron+neurons_per_layer[layer-1];\n\
                val += (inputs[index_neuron]*inputs[index_w]/scaling_factor);\n\
            }\n\
            val += inputs[index_b];\n\
            inputs[index_b] = activation(val);\n\
        }\n\
    }\n'
    # for layer in range(1,len(network)):
    #     noir_code+='\n    let mut layer = '+str(layer)+';\n\
    # index_layer += neurons_per_layer[layer - 1] * (neurons_per_layer[layer]+1);\n\
    # for i in 0 .. neurons_per_layer[layer] {\n\
    #     let mut val = 0;\n\
    #     let mut index_i = index_layer + i;\n\
    #     let mut index_j = index_layer - neurons_per_layer[layer] * neurons_per_layer[layer - 1] - neurons_per_layer[layer - 1];\n\
    #     for j in 0 .. neurons_per_layer[layer-1] {\n\
    #         val += (inputs[index_j+j]*inputs[index_j+j+neurons_per_layer[layer-1]]/scaling_factor);\n\
    #     }\n\
    #     std::println([index_i]);\n\
    #     val += inputs[index_i];\n\
    #     inputs[index_i] = val;\n\
    # }\n'
    noir_code += '    std::println([inputs['+str(num-1)+']]);\n\
}\n\
\n\
fn activation(x: u16) -> u16 {\n\
    let mut result = 0;\n\
    if x>result {\n\
        result = x;\n\
    }\n\
    result\n\
}\n\
\n\
#[test]\n\
fn test_main() {\n\
    let inputs:[u16; num] = '+str(inputs)+';\n\
    main(inputs);\n\
}'
    prover_str = 'inputs = '+str_inputs+'\n'
    with open(os.path.join(path, "src/main.nr"), "w+") as file:
        file.write(noir_code)
    with open(os.path.join(path, "Prover.toml"), "w+") as file:
        file.write(prover_str)


if __name__=="__main__":
    # network=[1024,512,2]  # 失败  Downloading the Ignite SRS (134.0 MB)
    # network=[60,40,2] # 成功 Downloading the Ignite SRS (786.4 KB)
    # network=[600,40,2]  # 成功 Downloading the Ignite SRS (7.3 MB)
    # network=[600,400,2]  # 失败 Downloading the Ignite SRS (58.7 MB)
    network=[712,712,2]  # 失败 Downloading the Ignite SRS (125.8 MB)
    # fnn_input_count(network)
    # fnn_inputs_value(network)
    create_noir_code(network, path='/mnt/code/noir_project/fnn3')
