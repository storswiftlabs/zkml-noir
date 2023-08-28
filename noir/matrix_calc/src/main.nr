use dep::std::println;

fn add<M>(matrix1: [Field; M], matrix2: [Field; M]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..matrix1.len(){
        result[i] = (matrix1[i] + matrix2[i]);
    }
    result
}

fn sub<M>(matrix1: [Field; M], matrix2: [Field; M]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..matrix1.len(){
        result[i] = (matrix1[i] - matrix2[i]);
    }
    result
}

fn hadamard_product<M>(matrix1: [Field; M], matrix2: [Field; M]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..matrix1.len(){
        result[i] = (matrix1[i] * matrix2[i]);
    }
    result
}

fn mul<M, N>(matrix1: [Field; M], matrix2: [Field; M], matrix1_shape: [Field; 2], matrix2_shape: [Field; 2], mut result: [Field; N]) -> [Field; N]{
    assert(matrix1_shape[1]==matrix2_shape[0]);
    for i in 0..matrix1_shape[0]{
        for j in 0..matrix2_shape[1]{
            let mut element = 0;
            for k in 0..matrix1_shape[1]{
                element += matrix1[i * matrix1_shape[1] + k] * matrix2[k * matrix2_shape[1] + j];
            }
            result[i*matrix2_shape[1]+j] = element;
        }
    }
    result
}

fn scalar_mul<M>(matrix: [Field; M], scalar: Field) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..matrix.len(){
        result[i] = scalar*matrix[i];
    }
    result
}

fn scalar_div<M>(matrix: [Field; M], scalar: Field) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..matrix.len(){
        result[i] = matrix[i]/scalar;
    }
    result
}

fn transpose<M>(matrix: [Field; M], shape: [Field; 2]) -> [Field; M]{
    let mut result = [0; M];
    for i in 0..shape[0]{
        for j in 0..shape[1]{
            result[j * shape[0] + i] = matrix[i * shape[1] + j];
        }
    }
    result
}

fn get_minor<M, N>(matrix: [Field; M], row: Field, col: Field, shape: [Field; 2], mut minor: [Field; N]) -> [Field; N]{
    let mut r: Field = 0;
    for i in 0..shape[0]{
        if i != row{
            r+=1;
            let mut c: Field = 0;
            for j in 0..shape[1]{
                if j != col{
                    c+=1;
                    minor[r*(shape[1]-1)+c]=(matrix[i * shape[1] + j]);
                }
            }
        }
    }
    minor
}

fn determinant<M>(matrix: [Field; M], shape: [Field; 2]) -> Field{
    assert(shape[0] == shape[1]);
    
    if shape[0] == 2{
        matrix[0] * matrix[3] - matrix[1] * matrix[2]
    }
    let mut det = 0;
    let mut sign = 1;
    for i in 0..shape[0]{
        let mut minor = [0; M];
        minor = get_minor(matrix, 0, i, shape, minor);
        det += sign * matrix[i] * determinant(minor, [shape[0] - 1, shape[1] - 1]);
        sign *= -1;
    }
    det
}

fn adjoint<M>(matrix: [Field; M], shape: [Field; 2]) -> [Field; M]{
    assert(shape[0] == shape[1]);
    
    let mut adjoint = [0; M];
    let mut sign = 1;
    for i in 0..shape[0]{
        for j in 0..shape[1]{
            let mut minor = [0; M];
            minor = get_minor(matrix, i, j, shape, minor);
            let mut cofactor = sign * determinant(minor, [shape[0] - 1, shape[1] - 1]);
            adjoint[i*shape[1]+j] = cofactor;
            sign *= -1;
        }
    }
    adjoint
}

fn inverse<M>(matrix: [Field; M], shape: [Field; 2]) -> [Field; M]{
    assert(shape[0] == shape[1]);
    
    let det = determinant(matrix, shape);
    assert(det != 0);
    
    let adjoint = adjoint(matrix, shape);
    
    let mut inverse = [0; M];
    for i in 0..shape[0]{
        for j in 0..shape[1]{
            inverse[i * shape[1] + j] = adjoint[i * shape[1] + j] / det;
        }
    }
    inverse
}

fn main() {
    
}

// #[test]
// fn test_main() {
//     add(1, 2);

//     // Uncomment to make test fail
//     // main(1, 1);
// }
