package autoregression

import (
    "encoding/json"
    "math"
    "reflect"
    "testing"
)

const (
    floatTolerance float64 = 0.000000000001
)

func TestNewModelOLS(t *testing.T) {
    data := []float64{2, 7, 1, -3, 2, -2}
    order := 2
    model, modelErr := NewModelOLS(data, order)
    if modelErr != nil {
        t.Errorf("Expected model but got error %v.", modelErr)
    }
    if model == nil {
        t.Fatal("Expected a model but got nil.")
    }
    expectedCoefficients := []float64{- float64(515) / float64(10302), - float64(1525) / float64(10302)}
    expectedNoise := - float64(31) / float64(202)
    if model.coefficients == nil {
        t.Errorf("Excpected coefficients %v but got nil", expectedCoefficients)
    }
    if len(model.coefficients) != len(expectedCoefficients) {
        t.Errorf("Excpected coefficients %v but got %v", expectedCoefficients, model.coefficients)
    }
    for index, value := range expectedCoefficients {
        if !floatsAreClose(value, model.coefficients[index], floatTolerance) {
            t.Errorf("Coefficient %d should be %.12f but is %.12f.", index, value, model.coefficients[index])
        }
    }
    if !floatsAreClose(model.noise, expectedNoise, floatTolerance) {
        t.Errorf("Expected noise of %.12f but got noise of %.12f.", expectedNoise, model.noise)
    }
    var expectedErrorVariance float64 = 0
    iterations := len(data) - order
    for iteration := 0; iteration < iterations; iteration++ {
        prediction := expectedNoise
        for coefficientIndex := order - 1; coefficientIndex >= 0; coefficientIndex-- {
            prediction += data[iteration + order - 1 - coefficientIndex] * expectedCoefficients[coefficientIndex]
        }
        predictionError := prediction - data[iteration + order]
        expectedErrorVariance += predictionError * predictionError / float64(iterations)
    }
    expectedStandardError := math.Sqrt(expectedErrorVariance)
    if !floatsAreClose(model.StandardError(), expectedStandardError, floatTolerance) {
        t.Errorf("Expected standard error of %.12f but got standard error of %.12f.", expectedStandardError, model.StandardError())
    }
}

func TestNewModelOLSNegativeOrder(t *testing.T) {
    data := []float64{2, 7, 1, -3, 2, -2}
    order := -1
    model, modelErr := NewModelOLS(data, order)
    if model != nil {
        t.Error("Expected nil model for negative order, but got non-nil model")
    }
    if modelErr != ErrNegativeOrder {
        t.Errorf("Expected ErrNegativeOrder error for negative order but got %v", modelErr)
    }
}

func TestNewModelOLSInsufficientData(t *testing.T) {
    data := []float64{2, 7, 1, -3, 2}
    order := 2
    model, modelErr := NewModelOLS(data, order)
    if model != nil {
        t.Error("Expected nil model for insufficient data, but got non-nil model")
    }
    if modelErr != ErrInsufficientDataForOrder {
        t.Errorf("Expected ErrInsufficientDataForOrder error for insufficient data but got %v", modelErr)
    }
}

func TestModelPrediction(t *testing.T) {
    data := []float64{2, 7, 1, -3, 2, -2}
    order := 2
    model, modelErr := NewModelOLS(data, order)
    if modelErr != nil {
        t.Errorf("Expected model but got error %v.", modelErr)
    }
    if model == nil {
        t.Fatal("Expected a model but got nil.")
    }
    expectedPredicion := model.noise
    newData := []float64{66, 88}
    for coeffIndex, coeff := range model.coefficients {
        expectedPredicion += newData[len(newData) - 1 - coeffIndex] * coeff
    }
    actualPrediction, predictionErr := model.Predict(newData)
    if predictionErr != nil {
        t.Fatalf("Expected prediction but got error %v.", predictionErr)
    }
    if !floatsAreClose(expectedPredicion, actualPrediction, floatTolerance) {
        t.Errorf("Expected predicion of %.12f but got %.12f.", expectedPredicion, actualPrediction)
    }
}

func TestModelPredictionWithZeroOrder(t *testing.T) {
    data := []float64{2, 7, 1, -3, 2, -2}
    order := 0
    model, modelErr := NewModelOLS(data, order)
    if modelErr != nil {
        t.Errorf("Expected model but got error %v.", modelErr)
    }
    if model == nil {
        t.Fatal("Expected a model but got nil.")
    }
    expectedPredicion := model.noise
    newData := []float64{}
    actualPrediction, predictionErr := model.Predict(newData)
    if predictionErr != nil {
        t.Fatalf("Expected prediction but got error %v.", predictionErr)
    }
    if !floatsAreClose(expectedPredicion, actualPrediction, floatTolerance) {
        t.Errorf("Expected predicion of %.12f but got %.12f.", expectedPredicion, actualPrediction)
    }
}

func TestJsonEncoding(t *testing.T) {
    firstModel := ARModel{coefficients: []float64{-0.5916191048362872,0.49113848002403127},
                          noise: -4.069465304896365,
                          standardError: 0.8622559253870782,
    }
    jsonBytes, marshalErr := json.Marshal(&firstModel)
    if marshalErr != nil {
        t.Fatalf("Error marshalling model to JSON: %v.\n", marshalErr)
    }
    var modelFromJson ARModel
    unmarshalErr := json.Unmarshal(jsonBytes, &modelFromJson)
    if unmarshalErr != nil {
        t.Fatalf("Error unmarshalling JSON to ARModel: %v.\n", unmarshalErr)
    }
    if !reflect.DeepEqual(modelFromJson, firstModel) {
        t.Errorf("Unmarshalled JSON differs from model. Expected %v but got %v.\n", firstModel, modelFromJson)
    }
}

func floatsAreClose(f1, f2, tolerance float64) bool {
    return math.Abs(f2 - f1) <= tolerance
}

