from transpiler.context.noir_context import NoirContext
from transpiler.core_module.control_pod import IfControl
from transpiler.core_module.for_loop_pod import ForLoop
from transpiler.core_module.statement_pod import Let
from transpiler.sub_module.primitive_type import UINT32, custom_type
from transpiler.sub_module.sign import (
    LEFT_BRACKET,
    RIGHT_BRACKET,
    SEMICOLON,
    LESS_THAN,
    LEFT_PARENTHESIS,
    RIGHT_PARENTHESIS,
    ADD,
    SUB,
    ASSIGNMENT,
    LEFT_BRACE,
    RIGHT_BRACE,
    COMMA,
    MULTI
)
from transpiler.utils.utils import table_format_control


def generate_k_means_noir_code(centers, display_type, fixed_number, noir_name: str = "main"):
    context = NoirContext()
    u32 = UINT32
    array_type = f"{LEFT_BRACKET}{display_type}{SEMICOLON}{len(centers[0])}{RIGHT_BRACKET}"
    body = []

    # Euclidean distance
    obtainEuclideanDistance = 'obtainEuclideanDistance'
    point = 'point'
    inputs = 'inputs'
    fn_inputs_type_and_name = {inputs: array_type, point: array_type}
    fn_result = display_type
    sum = 'sum'
    index = 'index'
    let = Let(sum, display_type, 0, True).get()
    for_loop = ForLoop(index, '0', '5',
                       f"{sum} "
                       f"{ADD}{ASSIGNMENT} "
                       f"{LEFT_PARENTHESIS}"
                       f"{point}{LEFT_BRACKET}{index}{RIGHT_BRACKET}"
                       f" {SUB} "
                       f"{inputs}{LEFT_BRACKET}{index}{RIGHT_BRACKET}"
                       f"{RIGHT_PARENTHESIS}"
                       f" {MULTI} "
                       f"{LEFT_PARENTHESIS}"
                       f"{point}{LEFT_BRACKET}{index}{RIGHT_BRACKET}"
                       f" {SUB} "
                       f"{inputs}{LEFT_BRACKET}{index}{RIGHT_BRACKET}"
                       f"{RIGHT_PARENTHESIS}"
                       f"{SEMICOLON}"
                       ).get()
    body.append(let)
    body.append(for_loop)
    body.append(sum)
    context.add_function(obtainEuclideanDistance, fn_inputs_type_and_name, fn_result, body)

    body = []

    # check_min
    check_min = 'check_min'
    fn_inputs_type_and_name = {}
    for index in range(0, len(centers)):
        fn_inputs_type_and_name['e' + str(index)] = display_type
    u3 = custom_type('u3')
    output = 'output'
    temp = 'temp'
    sign = LESS_THAN
    let_output = Let(output, u3, '0', True).get()
    body.append(let_output)
    let_temp = Let(temp, display_type, 'e0', True).get()
    body.append(let_temp)
    for index in range(1, len(fn_inputs_type_and_name)):
        body.append(IfControl('e' + str(index), temp, sign,
                              f"{temp} {ASSIGNMENT} {'e' + str(index)}{SEMICOLON}\n"
                              f"{output} {ASSIGNMENT} {str(index)}{SEMICOLON}"
                              ).get())
    body.append(output)
    context.add_function(check_min, fn_inputs_type_and_name, u3, body)

    body = []

    # main
    main = 'main'
    fn_inputs_type_and_name = {inputs: array_type}
    for index in range(0, len(centers)):
        fn_inputs_type_and_name['point'+str(index)] = array_type

    for index in range(0, len(centers)):
        body.append(Let(
            'e' + str(index),
            display_type,
            f"{obtainEuclideanDistance}{LEFT_PARENTHESIS}{inputs}{COMMA} {point + str(index)}{RIGHT_PARENTHESIS}",
            False
        ).get())
    body.append(f"{check_min}{LEFT_PARENTHESIS}{'e0,e1,e2,e3'}{RIGHT_PARENTHESIS}")
    context.add_function(main, fn_inputs_type_and_name, u3, body)

    data_arr = context.generate_noir_code_list(noir_name, fixed_number)
    return table_format_control(data_arr)


# def generate_main_inputs_type_and_name(centers, display_type):
#     row = len(centers[0])
#     col = len(centers)
#     array_name = 'Axis'
#     inputs_type_and_name = {
#         array_name: f"{LEFT_BRACKET}{display_type}{row}{RIGHT_BRACKET}",
#         'KMeansCenters': f"{LEFT_BRACKET}{array_name}{col}{RIGHT_BRACKET}"
#     }
#     return inputs_type_and_name
