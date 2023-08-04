// Code generated from Python2Noir
// Fixed number is 100

fn obtainEuclideanDistance(inputs : [i64;6],point : [i64;6],) -> pub i64 {
	let mut sum: i64 = 0;
	for index in 0..5 {
		sum += (point[index] - inputs[index]) * (point[index] - inputs[index]);
	}
	sum
}
fn check_min(e0 : i64,e1 : i64,e2 : i64,e3 : i64,) -> pub u3 {
	let mut output: u3 = 0;
	let mut temp: i64 = e0;
	if e1 < temp {
		temp = e1;
		output = 1;
	}
	if e2 < temp {
		temp = e2;
		output = 2;
	}
	if e3 < temp {
		temp = e3;
		output = 3;
	}
	output
}
fn main(inputs : [i64;6],point0 : [i64;6],point1 : [i64;6],point2 : [i64;6],point3 : [i64;6],) -> pub u3 {
	let e0: i64 = obtainEuclideanDistance(inputs, point0);
	let e1: i64 = obtainEuclideanDistance(inputs, point1);
	let e2: i64 = obtainEuclideanDistance(inputs, point2);
	let e3: i64 = obtainEuclideanDistance(inputs, point3);
	check_min(e0,e1,e2,e3)
}


#[test]
fn test_main() {
    let inputs: [i64;6] = [3550, 0, 100, 0, 0, 0];
    let p1: [i64;6] = [4056, 0, 60, 60, 0, 60];
    let p2: [i64;6] = [3732, 0, 13, 100, 65, 56];
    let p3: [i64;6] = [4062, 99, 100, 65, 99, 31];
    let p4: [i64;6] = [3687, 0, 100, 0, 0, 0];
    main(inputs, p1, p2, p3, p4);

    // Uncomment to make test fail
    // main(1, 1);
}
