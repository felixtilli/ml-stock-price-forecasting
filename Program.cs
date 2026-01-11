using Alpaca.Markets;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.TimeSeries;
using MLStockPriceForecasting.Models;
using System;

async Task<List<Candle>> GetCandles()
{
    var config = new ConfigurationBuilder().AddJsonFile(Path.Combine(AppContext.BaseDirectory, "appsettings.json"), optional: false, reloadOnChange: true).Build();
    var alpacaClient = Environments.Paper.GetAlpacaDataClient(new SecretKey(config["AlpacaApiKeyId"], config["AlpacaApiSecretKey"]));
    var historicalBars = await alpacaClient.GetHistoricalBarsAsync(new HistoricalBarsRequest("LODE", DateTime.Now.AddYears(-1), DateTime.Now.AddMinutes(-16), BarTimeFrame.Day));
    return historicalBars.Items.FirstOrDefault().Value.Select(Candle.ConvertIBarToCandle).ToList();
}

async Task<float> ForecastBySsa(List<Candle> candles)
{
    var mlContext = new MLContext();

    var pipeline = mlContext.Forecasting.ForecastBySsa(
        outputColumnName: nameof(SsaForecastOutput.Values),
        inputColumnName: nameof(Candle.Close),
        windowSize: 3,
        seriesLength: candles.Count,
        trainSize: candles.Count,
        horizon: 1);

    var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(candles));
    var engine = model.CreateTimeSeriesEngine<Candle, SsaForecastOutput>(mlContext);
    var result = engine.Predict().Values.FirstOrDefault();

    return result;
}

async Task<float> ForecastByRegressionLag(List<Candle> candles)
{
    var mlContext = new MLContext();

    var data = candles.Select((c, i) => new FeatureCandle
    {
        Close = c.Close,
        CloseLag1 = i > 0 ? candles[i - 1].Close : 0,
        CloseLag2 = i > 1 ? candles[i - 2].Close : 0,
        CloseLag3 = i > 2 ? candles[i - 3].Close : 0
    }).Skip(3).ToList();

    var mlData = mlContext.Data.LoadFromEnumerable(data);

    var pipeline = mlContext.Transforms
       .Concatenate("Features", "CloseLag1", "CloseLag2", "CloseLag3")
       .Append(mlContext.Regression.Trainers.Sdca(
           labelColumnName: "Close",
           featureColumnName: "Features"));

    var model = pipeline.Fit(mlData);
    var engine = mlContext.Model.CreatePredictionEngine<FeatureCandle, ForecastOutput>(model);

    var last = candles.TakeLast(3).ToArray();
    var nextPrediction = engine.Predict(new FeatureCandle
    {
        CloseLag1 = last[0].Close,
        CloseLag2 = last[1].Close,
        CloseLag3 = last[2].Close
    });

    return nextPrediction.Score;
}

async Task<float> ForecastByTree(List<Candle> candles)
{
    var mlContext = new MLContext();

    var data = candles.Select((c, i) => new FeatureCandle
    {
        Close = c.Close,
        CloseLag1 = i > 0 ? candles[i - 1].Close : 0,
        CloseLag2 = i > 1 ? candles[i - 2].Close : 0,
        CloseLag3 = i > 2 ? candles[i - 3].Close : 0
    }).Skip(3).ToList();

    var mlData = mlContext.Data.LoadFromEnumerable(data);

    var pipeline = mlContext.Transforms
        .Concatenate("Features", "CloseLag1", "CloseLag2", "CloseLag3")
        .Append(mlContext.Regression.Trainers.FastTree(
            labelColumnName: "Close",
            featureColumnName: "Features",
            numberOfLeaves: 20,
            numberOfTrees: 100,
            minimumExampleCountPerLeaf: 5));

    var model = pipeline.Fit(mlData);
    var engine = mlContext.Model.CreatePredictionEngine<FeatureCandle, ForecastOutput>(model);

    var last = candles.TakeLast(3).ToArray();
    var nextPrediction = engine.Predict(new FeatureCandle
    {
        CloseLag1 = last[0].Close,
        CloseLag2 = last[1].Close,
        CloseLag3 = last[2].Close
    });

    return nextPrediction.Score;
}

string FormatResult(string methodName, float prediction, float previousValue)
{
    return $"{methodName}: {Math.Round(prediction, 2).ToString("F2")} ({(prediction > previousValue ? "+" : "-")}). ";
}

async Task Test()
{
    var candles = await GetCandles();
    var trainCandles = candles.Take(candles.Count - 10).ToList();
    var testCandles = candles.Skip(candles.Count - 10).ToList();

    foreach (var testCandle in testCandles)
    {
        var ssaResult = await ForecastBySsa(trainCandles);
        var regressionLagResult = await ForecastByRegressionLag(trainCandles);
        var treeResult = await ForecastByTree(trainCandles);
        var lastValue = trainCandles.LastOrDefault().Close;

        Console.WriteLine(
            FormatResult("SSA", ssaResult, lastValue) +
            FormatResult("RegressionLag", regressionLagResult, lastValue) +
            FormatResult("Tree", treeResult, lastValue) +
            FormatResult("Actual", testCandle.Close, lastValue)
        );

        trainCandles.Add(testCandle);
    }
}

async Task Predict()
{
    var candles = await GetCandles();
    var ssaResult = await ForecastBySsa(candles);
    var regressionLagResult = await ForecastByRegressionLag(candles);
    var treeResult = await ForecastByTree(candles);
    var combinedResult = (ssaResult + regressionLagResult + treeResult) / 3;
    Console.WriteLine(FormatResult("Combined", combinedResult, candles.LastOrDefault().Close));
}

await Test();
Console.ReadLine();
