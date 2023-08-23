// Code generated from Python2Noir
use dep::std::println;
mod post_quantization_operation;
global point_scale : Field = 255;
global point_zero_point : Field = 0;
global point0 : [Field;6] = [139, 64, 191, 0, 64, 0];
global point1 : [Field;6] = [194, 76, 255, 255, 77, 255];
global point2 : [Field;6] = [129, 85, 85, 255, 170, 0];
global point3 : [Field;6] = [60, 0, 0, 255, 255, 255];
fn obtainEuclideanDistance(inputs : [Field;6],inputs_scale : Field,inputs_zero_point : Field,point : [Field;6],) -> pub Field {
	let mut sum: Field = 0;
	for index in 0..5 {
		let x: Field = post_quantization_operation::sub(point[index], point_zero_point, point_scale, inputs[index], inputs_zero_point, inputs_scale, point_zero_point, point_scale);
		let x_double: Field = post_quantization_operation::mul(x, point_zero_point, point_scale, x, point_zero_point, point_scale, point_zero_point, point_scale);
		sum = post_quantization_operation::add(sum, point_zero_point, point_scale, x_double, point_zero_point, point_scale, point_zero_point, point_scale);
	}
	sum = sum * 255;
	sum = post_quantization_operation::mul(sum, point_zero_point ,point_scale ,1 ,0 ,1 ,0 ,1);
	sum
}
fn check_min(e0 : Field,e1 : Field,e2 : Field,e3 : Field,) -> pub u3 {
	let mut output: u3 = 0;
	let mut temp: Field = e0;
	if e1 as u32 / 255 < temp as u32 / 255 {
		temp = e1;
		output = 1;
	}
	if e2 as u32 / 255 < temp as u32 / 255 {
		temp = e2;
		output = 2;
	}
	if e3 as u32 / 255 < temp as u32 / 255 {
		temp = e3;
		output = 3;
	}
	output
}
fn main(inputs : [Field;6],inputs_scale : Field,inputs_zero_point : Field,) -> pub u3 {
	let e0: Field = obtainEuclideanDistance(inputs, inputs_scale, inputs_zero_point, point0);
	let e1: Field = obtainEuclideanDistance(inputs, inputs_scale, inputs_zero_point, point1);
	let e2: Field = obtainEuclideanDistance(inputs, inputs_scale, inputs_zero_point, point2);
	let e3: Field = obtainEuclideanDistance(inputs, inputs_scale, inputs_zero_point, point3);
	check_min(e0,e1,e2,e3)
}



#[test]
fn test_k_means() {
    // let inputs:[Field;6] = [17, 0, 0, 255, 255, 255];
    let inputs:[Field;6] = [0, 0, 255, 0, 0, 0];
    let inputs_scale:Field = 255;
    let inputs_zero_point:Field = 0;
    let res = main(inputs, inputs_scale, inputs_zero_point);
    println(res);
    // assert(res==0);
    // Uncomment to make test fail
    // main(1, 1);
}
