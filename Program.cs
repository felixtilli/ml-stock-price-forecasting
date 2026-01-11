using Alpaca.Markets;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using MLStockPriceForecasting.Models;

var config = new ConfigurationBuilder().AddJsonFile(Path.Combine(AppContext.BaseDirectory, "appsettings.json"), optional: false, reloadOnChange: true).Build();
var alpacaClient = Environments.Paper.GetAlpacaDataClient(new SecretKey(config["AlpacaApiKeyId"], config["AlpacaApiSecretKey"]));
var historicalBars = await alpacaClient.GetHistoricalBarsAsync(new HistoricalBarsRequest("LODE", DateTime.Now.AddDays(-30), DateTime.Now.AddMinutes(-16), BarTimeFrame.Day));
var candles = historicalBars.Items.FirstOrDefault().Value.Select(Candle.ConvertIBarToCandle).ToList();

var mlContext = new MLContext();

var pipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: nameof(ForecastOutput.Values),
    inputColumnName: nameof(Candle.Close),
    windowSize: 3,
    seriesLength: candles.Count,
    trainSize: candles.Count,
    horizon: 1);

var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(candles));
var engine = model.CreateTimeSeriesEngine<Candle, ForecastOutput>(mlContext);
var result = engine.Predict().Values.FirstOrDefault();

Console.WriteLine("Forecasted value: " + result);
Console.ReadLine();
