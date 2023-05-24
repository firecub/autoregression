# autoregression
This is a package for generating autoregressive models for sequential or timeseries data.

## `NewModelOLS(data []float64, order int) (*ARModel, error)`
This function attempts to generate an autoregressive model using the method of Ordinary Least Squares. It requires some past data, as a `float64` slice, and an order (or lag). If successful, a model will be returned, which can then be used to make predictions.

## `ARModel.Order() int`
This method returns the order of the model.

## `ARModel.Predict(newData []float64) (float64, error)`
This method returns a prediction of the next value following a given it's preceding data of length `order`.

