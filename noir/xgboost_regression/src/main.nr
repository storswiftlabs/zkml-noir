// Code generated from Python2Noir
// inputs quantization scale reciprocal: 255
// inputs quantization zero-point: 0
// quantize_type: uint8
use dep::std::println;
mod quantize_arithmetic;
fn trees0class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			109
		} else {
			if inputs[0] <= 104 {
				146
			} else {
				181
			}
		}
	} else {
		if inputs[3] <= 128 {
			179
		} else {
			216
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees1class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			114
		} else {
			if inputs[0] <= 104 {
				140
			} else {
				166
			}
		}
	} else {
		if inputs[3] <= 128 {
			165
		} else {
			191
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees2class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			118
		} else {
			if inputs[0] <= 104 {
				136
			} else {
				155
			}
		}
	} else {
		if inputs[3] <= 128 {
			155
		} else {
			173
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees3class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			120
		} else {
			if inputs[0] <= 104 {
				134
			} else {
				147
			}
		}
	} else {
		if inputs[3] <= 128 {
			147
		} else {
			160
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees4class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			122
		} else {
			if inputs[0] <= 104 {
				132
			} else {
				141
			}
		}
	} else {
		if inputs[3] <= 128 {
			142
		} else {
			151
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees5class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			124
		} else {
			if inputs[0] <= 104 {
				130
			} else {
				137
			}
		}
	} else {
		if inputs[3] <= 128 {
			138
		} else {
			144
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees6class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			125
		} else {
			if inputs[0] <= 104 {
				129
			} else {
				134
			}
		}
	} else {
		if inputs[3] <= 128 {
			135
		} else {
			140
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees7class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			125
		} else {
			if inputs[0] <= 104 {
				129
			} else {
				132
			}
		}
	} else {
		if inputs[3] <= 128 {
			133
		} else {
			136
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees8class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			126
		} else {
			if inputs[0] <= 104 {
				128
			} else {
				131
			}
		}
	} else {
		if inputs[3] <= 128 {
			131
		} else {
			134
		}
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn trees9class0(inputs : [u32;6],) -> pub u32 {
	if inputs[1] <= 128 {
		if inputs[3] <= 128 {
			126
		} else {
			if inputs[0] <= 104 {
				128
			} else {
				130
			}
		}
	} else {
		131
	}
	// inputs quantization scale reciprocal: 128, zero-point: 127

}
fn count_prob<N>(class_prob : [u32;N],) -> pub u32 {
	let mut cls: Field = class_prob[0] as Field;
	for i in 1..N {
		cls = quantize_arithmetic::add(cls,127,128,class_prob[i] as Field,127,128,127,128);
	}
	cls = quantize_arithmetic::mul(cls,127,128,1,0,1,125,13);
	((cls*128) as u32)/128
}
fn main(inputs : [u32;6],) -> pub u32 {
	let trees0_class0: u32 = trees0class0(inputs);
	let trees1_class0: u32 = trees1class0(inputs);
	let trees2_class0: u32 = trees2class0(inputs);
	let trees3_class0: u32 = trees3class0(inputs);
	let trees4_class0: u32 = trees4class0(inputs);
	let trees5_class0: u32 = trees5class0(inputs);
	let trees6_class0: u32 = trees6class0(inputs);
	let trees7_class0: u32 = trees7class0(inputs);
	let trees8_class0: u32 = trees8class0(inputs);
	let trees9_class0: u32 = trees9class0(inputs);
	let res: u32 = count_prob([trees0_class0, trees1_class0, trees2_class0, trees3_class0, trees4_class0, trees5_class0, trees6_class0, trees7_class0, trees8_class0, trees9_class0]);
	res
}


#[test]
fn test_main_0() {
    let inputs: [u32;6] = [0, 0, 255, 0, 0, 0];
    let res = main(inputs);
    println(res);
    assert(res==0x76);
}

#[test]
fn test_main_1() {
    let inputs: [u32;6] = [17, 0, 0, 255, 255, 255];
    let res =  main(inputs);
    println(res);
    assert(res==0x83);
}

#[test]
fn test_main_2() {
    let inputs: [u32;6] = [255, 0, 255, 255, 0, 255];
    let res = main(inputs);
    println(res);
    assert(res==0x8f);
}

#[test]
fn test_main_3() {
    let inputs: [u32;6] = [247, 255, 255, 255, 255, 0];
    let res = main(inputs);
    println(res);
    assert(res==0x9c);
}
