import math

from transpiler.context.noir_context import NoirContext
from transpiler.core_module.control_pod import IfControl
from transpiler.core_module.for_loop_pod import ForLoop
from transpiler.core_module.statement_pod import Let
from transpiler.sub_module.primitive_type import UINT32, custom_type, FIELD
from transpiler.sub_module.sign import (
    LEFT_BRACKET,
    RIGHT_BRACKET,
    SEMICOLON,
    LESS_THAN,
    LEFT_PARENTHESIS,
    RIGHT_PARENTHESIS,
    ASSIGNMENT,
    COMMA,
)
from transpiler.utils.utils import table_format_control

from quantization.quantize import get_max, get_min, calc_scale, calc_zero_point, quantize_not_clip


def generate_k_means_noir_code(centers, scale, zero_point, display_type, noir_name: str = "main"):

    u32 = UINT32
    array_type = f"{LEFT_BRACKET}{display_type}{SEMICOLON}{len(centers[0])}{RIGHT_BRACKET}"
    point = 'point'
    inputs = 'inputs'

    context = NoirContext()
    mod_name = "post_quantization_operation"
    context.add_mod(mod_name)
    context.add_use("dep::std::println")

    # add global variable
    point_scale = 'point_scale'
    context.add_global((point_scale, FIELD, scale))
    point_zero_point = 'point_zero_point'
    context.add_global((point_zero_point, FIELD, zero_point))
    for i in range(0, len(centers)):
        context.add_global((point+str(i), array_type, centers[i]))

    body = []

    # Euclidean distance
    obtainEuclideanDistance = 'obtainEuclideanDistance'
    inputs_scale = 'inputs_scale'
    inputs_zero_point = 'inputs_zero_point'
    fn_inputs_type_and_name = {inputs: array_type, inputs_scale: FIELD, inputs_zero_point: FIELD, point: array_type}
    fn_result = display_type
    sum = 'sum'
    index = 'index'
    let = Let(sum, display_type, 0, True).get()

    x = 'x'
    let_x = Let(x, FIELD,
                f"{mod_name}::sub{LEFT_PARENTHESIS}"
                    f"{point}{LEFT_BRACKET}{index}{RIGHT_BRACKET}{COMMA} "
                    f"{point_zero_point}{COMMA} {point_scale}{COMMA} "
                    f"{inputs}{LEFT_BRACKET}{index}{RIGHT_BRACKET}{COMMA} "
                    f"{inputs_zero_point}{COMMA} {inputs_scale}{COMMA} "
                    f"{point_zero_point}{COMMA} {point_scale}"
                f"{RIGHT_PARENTHESIS}",
                False).get()
    x_double = 'x_double'
    let_x_double = Let(x_double, FIELD,
                       f"{mod_name}::mul{LEFT_PARENTHESIS}"
                           f"{x}{COMMA} {point_zero_point}{COMMA} {point_scale}{COMMA} "
                           f"{x}{COMMA} {point_zero_point}{COMMA} {point_scale}{COMMA} "
                           f"{point_zero_point}{COMMA} {point_scale}"
                       f"{RIGHT_PARENTHESIS}",
                       False).get()

    obtainEuclideanDistance_body = [let_x, let_x_double, f"{sum} {ASSIGNMENT} {mod_name}::sub{LEFT_PARENTHESIS}"
                                                         f"{sum}{COMMA} {point_zero_point}{COMMA} {point_scale}{COMMA} "
                                                         f"{x_double}{COMMA} {point_zero_point}{COMMA} "
                                                         f"{point_scale}{COMMA} "
                                                         f"{point_zero_point}{COMMA} {point_scale}"
                                                         f"{RIGHT_PARENTHESIS}{SEMICOLON}"]

    for_loop = ForLoop(index, '0', '5', '\n'.join(obtainEuclideanDistance_body)).get()
    body.append(let)
    body.append(for_loop)
    body.append(sum+' * 255')
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
        body.append(IfControl(f"{'e'}{str(index)} as {u32} / {'255'}", f"{temp} as {u32} / {'255'}", sign,
                              f"{temp} {ASSIGNMENT} {'e' + str(index)}{SEMICOLON}\n"
                              f"{output} {ASSIGNMENT} {str(index)}{SEMICOLON}"
                              ).get())
    body.append(output)
    context.add_function(check_min, fn_inputs_type_and_name, u3, body)

    body = []

    # main
    main = 'main'
    fn_inputs_type_and_name = {inputs: array_type, inputs_scale: FIELD, inputs_zero_point: FIELD}
    for index in range(0, len(centers)):
        body.append(Let(
            'e' + str(index),
            display_type,
            f"{obtainEuclideanDistance}{LEFT_PARENTHESIS}{inputs}{COMMA} {inputs_scale}{COMMA} {inputs_zero_point}{COMMA} {point + str(index)}{RIGHT_PARENTHESIS}",
            False
        ).get())
    body.append(f"{check_min}{LEFT_PARENTHESIS}{'e0,e1,e2,e3'}{RIGHT_PARENTHESIS}")
    context.add_function(main, fn_inputs_type_and_name, u3, body)

    data_arr = context.generate_noir_code_list()
    return table_format_control(data_arr)


def quantize_centers(centers, _type):
    x = [element for sublist in centers for element in sublist]

    _f_min, _f_max = get_min(x), get_max(x)
    print(_f_min, _f_max)
    _scale_molecule, _scale_denominator = calc_scale(x, _type)
    scale = math.ceil(_scale_molecule / _scale_denominator)
    _zero_point = calc_zero_point(x, scale, _type)
    quantized_centers = quantize_not_clip(x, scale, _zero_point)
    centers = [quantized_centers[i:i + len(centers[0])] for i in range(0, len(quantized_centers), len(centers[0]))]
    return _scale_molecule, _scale_denominator, _zero_point, centers


def generate_inputs(centers):
    inputs_name_and_type = ''
    for row in range(0, len(centers)):
        temp = f"{'point'}{str(row)} = {'['}"
        for col in range(0, len(centers[0])):
            temp += str(centers[row][col])
            if col != len(centers[0]) - 1:
                temp += ','
        temp += ']\n'
        inputs_name_and_type += temp
    return inputs_name_and_type
