use dep::std::println;


// quantized arithmetic mul
// (x_scale_reciprocal * (x_quantized - x_zero_point)) * (y_scale_reciprocal * (y_quantized - y_zero_point)) / z_scale_reciprocal + z_zero_point;
fn mul(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let z_quantized =
       1 / (x_scale_reciprocal * y_scale_reciprocal / z_scale_reciprocal) *
       (x_quantized - x_zero_point) *
       (y_quantized - y_zero_point) +
       z_zero_point;
   z_quantized
}

// quantized arithmetic div
// (x_scale_reciprocal * (x_quantized - x_zero_point)) / (y_scale_reciprocal * (y_quantized - y_zero_point)) / z_scale_reciprocal + z_zero_point;
fn div(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let z_quantized =
       1 / ( x_scale_reciprocal / y_scale_reciprocal / z_scale_reciprocal) *
       (x_quantized - x_zero_point) /
       (y_quantized - y_zero_point) +
       z_zero_point;
   z_quantized
}

// quantized arithmetic add
// ((x_scale_reciprocal * (x_quantized - x_zero_point)) + (y_scale_reciprocal * (y_quantized - y_zero_point))) / z_scale_reciprocal + z_zero_point;
fn add(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let mut multiplier_x_y: Field = 0;
   let x_scale_100: u8 = (x_scale_reciprocal * 100) as u8;
   let y_scale_100: u8 = (y_scale_reciprocal * 100) as u8;
   if x_scale_100 > y_scale_100 {
       multiplier_x_y = 2 * y_scale_reciprocal;
   } else {
       multiplier_x_y = 2 * x_scale_reciprocal;
   }
   let multiplier_x = x_scale_reciprocal / multiplier_x_y;
   let multiplier_y = y_scale_reciprocal / multiplier_x_y;
   let multiplier_x_y_z = multiplier_x_y / z_scale_reciprocal;
   let z_quantized =
       multiplier_x_y_z *
       (multiplier_x * (x_quantized - x_zero_point) + multiplier_y * (y_quantized - y_zero_point)) +
       z_zero_point;
   z_quantized
}

// quantized arithmetic sub
// ((x_scale_reciprocal * (x_quantized - x_zero_point)) - (y_scale_reciprocal * (y_quantized - y_zero_point))) / z_scale_reciprocal + z_zero_point;
fn sub(x_quantized: Field, x_zero_point: Field, x_scale_reciprocal: Field, y_quantized: Field, y_zero_point: Field, y_scale_reciprocal: Field, z_zero_point: Field, z_scale_reciprocal: Field) -> Field{
   let mut multiplier_x_y: Field = 0;
   let x_scale_100: u8 = (x_scale_reciprocal * 100) as u8;
   let y_scale_100: u8 = (y_scale_reciprocal * 100) as u8;
   if x_scale_100 > y_scale_100 {
       multiplier_x_y = 2 * y_scale_reciprocal;
   } else {
       multiplier_x_y = 2 * x_scale_reciprocal;
   }
   let multiplier_x = x_scale_reciprocal / multiplier_x_y;
   let multiplier_y = y_scale_reciprocal / multiplier_x_y;
   let multiplier_x_y_z = multiplier_x_y / z_scale_reciprocal;
   let z_quantized =
       multiplier_x_y_z *
       (multiplier_x * (x_quantized - x_zero_point) - multiplier_y * (y_quantized - y_zero_point)) +
       z_zero_point;
   z_quantized
}
