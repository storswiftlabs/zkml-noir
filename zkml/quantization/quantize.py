import torch

INT = ["int8", "int16", "int32", "int64"]
UINT = ["uint8", "uint16", "uint32", "uint64"]
Q_MAX = "Q_MAX"
Q_MIN = "Q_MIN"


def get_uint_type_range(uint_type: str) -> int:
    res = ""
    for i in uint_type:
        if str.isdigit(i):
            res += i
    return int(res)


QUANTIZE_RANGE = {UINT[0]: {Q_MIN: 0, Q_MAX: (1 << get_uint_type_range(UINT[0])) - 1},
                  UINT[1]: {Q_MIN: 0, Q_MAX: (1 << get_uint_type_range(UINT[1])) - 1},
                  UINT[2]: {Q_MIN: 0, Q_MAX: (1 << get_uint_type_range(UINT[2])) - 1},
                  UINT[3]: {Q_MIN: 0, Q_MAX: (1 << get_uint_type_range(UINT[3])) - 1}}


def get_max(x: list):
    x_max = x[0]
    for ele in x:
        if x_max < ele:
            x_max = ele
    return x_max


def get_min(x: list):
    x_min = x[0]
    for ele in x:
        if x_min > ele:
            x_min = ele
    return x_min


def get_quantize_range(target_type):
    assert target_type in UINT or target_type in INT
    q_max = QUANTIZE_RANGE[target_type][Q_MAX]
    q_min = QUANTIZE_RANGE[target_type][Q_MIN]
    return q_min, q_max


def calc_scale(x: list, target_type) -> float:
    q_min, q_max = get_quantize_range(target_type)
    # Calculate value range (denominator)
    x_range = get_max(x) - get_min(x)
    x_range = 1 if x_range == 0 else x_range

    # Calculate scale
    scale_molecule = q_max - q_min
    scale_denominator = x_range
    return scale_molecule, scale_denominator


def calc_zero_point(x: list, scale_molecule, scale_denominator, target_type: str) -> int:
    q_min, q_max = get_quantize_range(target_type)
    # zero_point = (-scale_molecule / scale_denominator * get_min(x) + q_min).round()
    zero_point = (-scale_molecule / scale_denominator * get_max(x) + q_max).__round__()
    return zero_point


def quantize(values: list, q_scale, q_zero_point: int, target_type: str):
    # detect input arg "target_type" in uint or int list
    q_min, q_max = get_quantize_range(target_type)
    x_quantize = []
    for value in values:
        x_quantize.append((value * q_scale + q_zero_point).__round__())
        # x_quantize = torch.clip((values * q_scale + q_zero_point).round(), min=Q_MIN, max=Q_MAX)
    x_quantize = torch.clip(torch.tensor(x_quantize), min=q_min, max=q_max)

    return [int(x_ele) for x_ele in x_quantize]


def quantize_not_clip(values: list, q_scale, q_zero_point: int):
    # detect input arg "target_type" in uint or int list
    x_quantize = []
    for value in values:
        x_quantize.append((value * q_scale + q_zero_point).__round__())
        # x_quantize = torch.clip((values * q_scale + q_zero_point).round(), min=Q_MIN, max=Q_MAX)
    x_quantize = torch.tensor(x_quantize)

    return [int(x_ele) for x_ele in x_quantize]


def dequantize(x_quantize: list, q_scale, q_zero_point: int, target_type: str):
    # detect input arg "target_type" in uint or int list
    _, _ = get_quantize_range(target_type)
    x_dequantize = []
    for ele in x_quantize:
        x_dequantize.append((ele - q_zero_point) / q_scale)
    return x_dequantize


def quantize_all(values: list, target_type: str):
    # quantize args (q_scale, q_zero_point) calculate by list
    # detect input arg "target_type" in uint or int list
    q_min, q_max = get_quantize_range(target_type)
    scale_molecule, scale_denominator = calc_scale(values, target_type)
    q_zero_point = calc_zero_point(values, scale_molecule, scale_denominator, target_type)
    x_quantize = []
    for value in values:
        x_quantize.append((value * scale_molecule / scale_denominator + q_zero_point).__round__())
        # x_quantize = torch.clip((values * q_scale + q_zero_point).round(), min=Q_MIN, max=Q_MAX)
    x_quantize = torch.clip(torch.tensor(x_quantize), min=q_min, max=q_max)
    x_quantize = [int(x_ele) for x_ele in x_quantize]
    return x_quantize, scale_molecule, scale_denominator, q_zero_point, get_max(values), get_min(values)


if __name__ == '__main__':
    x = [-3.0, 0.1, 3.2, -3.0, -0.3, 3.2, -2.0, 0.2, 2.0, -1.0, 0.1, 1.0, -3.2, 1.0, 3.0]
    _type = "uint8"
    _f_min, _f_max = get_max(x), get_min(x)
    print(_f_max, _f_min)
    _scale_molecule, _scale_denominator = calc_scale(x, _type)
    _zero_point = calc_zero_point(x, _scale_molecule, _scale_denominator, _type)
    scale = _scale_molecule / _scale_denominator
    print("scale", scale)
    print("zero_point", _zero_point)
    print(quantize(x, scale, _zero_point, _type))
    print(dequantize(
        quantize(x, scale, _zero_point, _type),
        scale,
        _zero_point,
        _type))
