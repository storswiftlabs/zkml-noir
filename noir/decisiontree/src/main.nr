// Code generated from Python2Noir
// Fixed number is 10

fn main(inputs : [i32;6],) -> pub u3 {
	if inputs[2] < 5 {
		if inputs[3] < 5 {
			0
		} else {
			1
		}
	} else {
		if inputs[0] < 380 {
			0
		} else {
			if inputs[4] < 5 {
				2
			} else {
				if inputs[3] < 5 {
					2
				} else {
					3
				}
			}
		}
	}
}


#[test]
fn test_main() {
    let inputs: [i32;6] = [3550, 0, 100, 0, 0, 0];
    main(inputs);

    // Uncomment to make test fail
    // main(1, 1);
}
