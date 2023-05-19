package autoregression

import (
    "errors"
    "gonum.org/v1/gonum/mat"
    "math"
)

var (
    ErrNegativeOrder = errors.New("autoregression: model order cannot be negative")
    ErrInsufficientDataForOrder = errors.New("autoregression: the number of elements in the data must be greater then one more than twice the order")
    ErrSingularCovariantMatrix = errors.New("autoregression: the covariant matrix generated from the data was singular")
    ErrIncorrectDataLength = errors.New("autoregression: the length of the supplied new data was not equal to the order")
)

type (
    ARModel struct {
        coefficients []float64
        noise float64
        standardError float64
    }
    
    symmetricSquareMatrix struct {
       size int
       elements []float64
    }
    
    vector []float64
)

func newSymmetricSquareMatrix(size int) *symmetricSquareMatrix {
    //As the matrix is square, it is only necessary to store the elements on or above the diagonal.
    return &symmetricSquareMatrix {size, make([]float64, size * (size + 1) / 2)}
}

func (m *symmetricSquareMatrix) Dims() (int, int) {
    return m.size, m.size
}

func (m *symmetricSquareMatrix) At(i, j int) float64 {
    r, c := i, j
    if c < r {
        r, c = j, i
    }
    return m.elements[r * m.size - (r-1) * r / 2 + c - r]
}

func (m *symmetricSquareMatrix) T() mat.Matrix {
    return m
}

func (m *symmetricSquareMatrix) setElement(r, c int, value float64) {
    m.elements[r * m.size - (r-1) * r / 2 + c - r] = value
}

func (model *ARModel) Order() int {
    return len(model.coefficients)
}

func (model *ARModel) StandardError() float64 {
    return model.standardError
}

func (model *ARModel) Predict(newData []float64) (float64, error) {
    if len(newData) != len(model.coefficients) {
        return 0, ErrIncorrectDataLength
    }
    prediction := model.noise
    for dataIndex, dataValue := range newData {
        prediction += model.coefficients[len(model.coefficients) - 1 - dataIndex] * dataValue
    }
    return prediction, nil
}

func (v vector) Dims() (int, int) {
    return len(v), 1
}

func (v vector) At(i, _ int) float64 {
    return v[i]
}

func (v vector) T() mat.Matrix {
    return mat.TransposeVec{v}
}

func (v vector) AtVec(i int) float64 {
    return v[i]
}

func (v vector) Len() int {
    return len(v)
}

// Generate a new AR model using the method of least squares.
func NewModelOLS(data []float64, order int) (*ARModel, error) {
    if order < 0 {
        return nil, ErrNegativeOrder
    }
    if len(data) <= 2 * order + 1 {
        return nil, ErrInsufficientDataForOrder
    }
    cm := makeCovariantMatrix(data, order)
    cv := makeCovariantVector(data, order)
    var coefficientVector mat.VecDense
    if order > 0 {
        var lu mat.LU
        lu.Factorize(cm)
        solveErr := lu.SolveVecTo(&coefficientVector, false, cv)
        if solveErr != nil {
            return nil, ErrSingularCovariantMatrix
        }
    }
    coefficients := make([]float64, order)
    for coefficientIndex := 0; coefficientIndex < order; coefficientIndex++ {
        coefficients[coefficientIndex] = coefficientVector.AtVec(coefficientIndex)
    }
    var noise float64 = 0
    var deviationVariance float64 = 0
    iterations := len(data) - order
    for k := 0; k < iterations; k++ {
        var coeffProduct float64 = 0
        for j := 0; j < order; j++ {
            coeffProduct += coefficients[j] * data[len(data) - j - 2 - k]
        }
        coeffProductDeviation := coeffProduct - data[len(data) - 1 - k]
        deviationVariance += coeffProductDeviation * coeffProductDeviation / float64(iterations)
        noise += data[len(data) - 1 - k] / float64(iterations) - coeffProduct / float64(iterations)
    }
    standardError := math.Sqrt(deviationVariance - noise * noise)
    return &ARModel{coefficients, noise, standardError}, nil
}

func makeCovariantMatrix(data []float64, order int) mat.Matrix {
    m := newSymmetricSquareMatrix(order)
    for r := 0; r < order; r++ {
        for c := r; c < order; c++ {
            m.setElement(r, c, cov(data, r + 1, c + 1, order))
        }
    }
    return m
}

func makeCovariantVector(data []float64, order int) mat.Vector {
    var v vector = make([]float64, order)
    for r := 0; r < order; r++ {
        v[r] = cov(data, r + 1, 0, order)
    }
    return v
}

func cov(data []float64, r, c, order int) float64 {
    iterations := len(data) - order
    var rTally, cTally, rcTally float64 = 0, 0, 0
    for k := 0; k < iterations; k++ {
        rTally += data[len(data) - r - 1 - k]
        cTally += data[len(data) - c - 1 - k]
        rcTally += data[len(data) - r - 1 - k] * data[len(data) - c - 1 - k]
    }
    return rcTally - rTally * cTally / float64(iterations)
}

