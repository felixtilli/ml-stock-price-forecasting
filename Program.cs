using Alpaca.Markets;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.TimeSeries;
using MLStockPriceForecasting.Models;
using MLStockPriceForecasting.Projections;

async Task GetCandles(string symbol)
{
    Console.WriteLine("Fetching candles for " + symbol + "...");
    var now = DateTime.Now;
    var fromDate = now.AddYears(-10);
    var toDate = now.AddMinutes(-16);

    using (var db = new AppDbContext())
    {
        var latestValue = db.Values
            .Where(v => v.Stock.Symbol == symbol)
            .OrderByDescending(v => v.Date)
            .FirstOrDefault();

        if (latestValue != null)
        {
            fromDate = latestValue.Date.AddDays(1);
        }
    }

    if (fromDate >= now)
    {
        return;
    }

    var config = new ConfigurationBuilder().AddJsonFile(Path.Combine(AppContext.BaseDirectory, "appsettings.json"), optional: false, reloadOnChange: true).Build();
    var alpacaClient = Environments.Paper.GetAlpacaDataClient(new SecretKey(config["AlpacaApiKeyId"], config["AlpacaApiSecretKey"]));
    var result = new List<Candle>();

    var request = new HistoricalBarsRequest(symbol, fromDate, toDate, BarTimeFrame.Day);
    do
    {
        var page = await alpacaClient.ListHistoricalBarsAsync(request);

        result.AddRange(page.Items.Select(Candle.ConvertIBarToCandle).ToList());

        request.WithPageToken(page.NextPageToken);
    } while (request.Pagination.Token is not null);

    await SaveValues(symbol, result);
}

async Task SaveValues(string symbol, List<Candle> candles)
{
    using (var db = new AppDbContext())
    {
        var stock = db.Stocks.FirstOrDefault(s => s.Symbol == symbol);
        if (stock == null)
        {
            stock = new Stock { Symbol = symbol };
            db.Stocks.Add(stock);
            await db.SaveChangesAsync();
        }

        foreach (var candle in candles)
        {
            if (db.Values.Any(x => x.StockId == stock.Id && x.Date == candle.TimeUtc))
            {
                continue;
            }

            var value = new Value
            {
                StockId = stock.Id,
                Date = candle.TimeUtc,
                Open = candle.Open,
                High = candle.High,
                Low = candle.Low,
                Close = candle.Close,
                Volume = candle.Volume,
                TradeCount = candle.TradeCount,
                Vwap = candle.Vwap
            };
            db.Values.Add(value);
        }
        await db.SaveChangesAsync();
    }
}

async Task SaveForecast(string symbol, string methodName, float predictedValue, DateTime date)
{
    using (var db = new AppDbContext())
    {
        var forecastingStrategy = db.ForecastingStrategies.FirstOrDefault(fs => fs.Name == methodName);
        if (forecastingStrategy == null)
        {
            forecastingStrategy = new ForecastingStrategy { Name = methodName };
            db.ForecastingStrategies.Add(forecastingStrategy);
            await db.SaveChangesAsync();
        }

        var stock = db.Stocks.FirstOrDefault(s => s.Symbol == symbol);
        if (stock == null)
        {
            stock = new Stock { Symbol = symbol };
            db.Stocks.Add(stock);
            await db.SaveChangesAsync();
        }

        if (db.Forecasts.Any(x => x.StockId == stock.Id && x.Date == date && x.ForecastingStrategyId == forecastingStrategy.Id))
        {
            return;
        }

        var forecast = new Forecast
        {
            StockId = stock.Id,
            Date = date,
            ClosePrice = predictedValue,
            ForecastingStrategyId = forecastingStrategy.Id
        };

        db.Forecasts.Add(forecast);
        await db.SaveChangesAsync();
    }
}

async Task<float> ForecastBySsa(string symbol, DateTime date, List<Candle> candles)
{
    var strategyName = "SSA";

    using (var db = new AppDbContext())
    {
        var existingForecast = await db.Forecasts.FirstOrDefaultAsync(x =>
            x.Stock.Symbol == symbol &&
            x.ForecastingStrategy.Name == strategyName &&
            x.Date == date);

        if (existingForecast != null)
        {
            return existingForecast.ClosePrice;
        }
    }

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

    await SaveForecast(symbol, strategyName, result, date);

    return result;
}

async Task<float> ForecastByRegressionLag(string symbol, DateTime date, List<Candle> candles)
{
    var strategyName = "RegressionLag";

    using (var db = new AppDbContext())
    {
        var existingForecast = await db.Forecasts.FirstOrDefaultAsync(x =>
            x.Stock.Symbol == symbol &&
            x.ForecastingStrategy.Name == strategyName &&
            x.Date == date);

        if (existingForecast != null)
        {
            return existingForecast.ClosePrice;
        }
    }

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

    var result = nextPrediction.Score;

    await SaveForecast(symbol, strategyName, result, date);

    return result;
}

async Task<float> ForecastByTree(string symbol, DateTime date, List<Candle> candles)
{
    var strategyName = "Tree";

    using (var db = new AppDbContext())
    {
        var existingForecast = await db.Forecasts.FirstOrDefaultAsync(x =>
            x.Stock.Symbol == symbol &&
            x.ForecastingStrategy.Name == strategyName &&
            x.Date == date);

        if (existingForecast != null)
        {
            return existingForecast.ClosePrice;
        }
    }

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

    var result = nextPrediction.Score;

    await SaveForecast(symbol, "Tree", result, date);

    return result;
}

string FormatResult(string methodName, float prediction, float previousValue)
{
    return $"{methodName}: {Math.Round(prediction, 2).ToString("F2")} ({(prediction > previousValue ? "+" : "-")}). ";
}

async Task Predict(string symbol)
{
    Console.WriteLine("Predicting " + symbol + "...");
    var candles = new List<Candle>();
    var datesToPredict = new List<DateTime>();

    using (var db = new AppDbContext())
    {
        var stock = db.Stocks.FirstOrDefault(s => s.Symbol == symbol);

        candles = db.Values.Where(v => v.StockId == stock.Id).OrderBy(v => v.Date).Select(v => new Candle
        {
            TimeUtc = v.Date,
            Open = v.Open,
            High = v.High,
            Low = v.Low,
            Close = v.Close,
            Volume = v.Volume,
            TradeCount = v.TradeCount,
            Vwap = v.Vwap
        }).ToList();

        var lastForecast = db.Forecasts
            .Where(f => f.StockId == stock.Id)
            .OrderByDescending(f => f.Date)
            .FirstOrDefault();

        var lastForecastDate = lastForecast != null ? lastForecast.Date : candles.FirstOrDefault().TimeUtc;

        var lastValueDate = candles.LastOrDefault().TimeUtc;

        for (DateTime date = lastForecastDate; date <= lastValueDate.AddDays(1); date = date.AddDays(1))
        {
            if (date > lastForecastDate)
            {
                datesToPredict.Add(date);
            }
        }
    }

    foreach (var date in datesToPredict)
    {
        var relevantCandles = candles.Where(c => c.TimeUtc < date).ToList();

        if (relevantCandles.Count < 1000)
        {
            continue;
        }

        var ssaTask = ForecastBySsa(symbol, date, relevantCandles);
        var regressionLagTask = ForecastByRegressionLag(symbol, date, relevantCandles);
        var treeTask = ForecastByTree(symbol, date, relevantCandles);

        await Task.WhenAll(ssaTask, regressionLagTask, treeTask);

        var ssaResult = ssaTask.Result;
        var regressionLagResult = regressionLagTask.Result;
        var treeResult = treeTask.Result;
        var combinedResult = (ssaResult + regressionLagResult + treeResult) / 3;
        var lastValue = relevantCandles.LastOrDefault().Close;

        Console.WriteLine(
            date.ToString("yyyy-MM-dd") +
            ": " +
            FormatResult("SSA", ssaResult, lastValue) +
            FormatResult("RegressionLag", regressionLagResult, lastValue) +
            FormatResult("Tree", treeResult, lastValue) +
            FormatResult("Combined", combinedResult, lastValue)
        );
    }
}

async Task AnalyzePredicitons(string symbol)
{
    Console.WriteLine("Analyzing predictions for " + symbol + "...");

    using (var db = new AppDbContext())
    {
        var predicitons = await db.Forecasts
            .Where(x =>
                x.Stock.Symbol == symbol &&
                !db.ForecastResults.Any(fr => fr.ForecastId == x.Id)
            )
            .ToListAsync();

        if (!predicitons.Any())
        {
            return;
        }

        var values = await db.Values
            .Where(x =>
                x.Stock.Symbol == symbol
            )
            .ToListAsync();

        if (!values.Any())
        {
            return;
        }

        foreach (var prediction in predicitons)
        {
            var previousValue = values.FirstOrDefault(x => x.Date == prediction.Date);
            if (previousValue == null)
            {
                continue;
            }
            var nextValue = values.FirstOrDefault(x => x.Date == prediction.Date.AddDays(1));
            if (nextValue == null)
            {
                continue;
            }
            var predicitionWasPositive = prediction.ClosePrice > previousValue.Close;
            var outComeWasPositive = nextValue.Close > previousValue.Close;
            var forecastResult = new ForecastResult
            {
                ForecastId = prediction.Id,
                DirectionCorrect = (predicitionWasPositive && outComeWasPositive) || (!predicitionWasPositive && !outComeWasPositive),
                Diff = Math.Abs(prediction.ClosePrice - nextValue.Close)
            };
            db.ForecastResults.Add(forecastResult);
            await db.SaveChangesAsync();
        }
    }
}

var symbols = new List<string>();

using (var db = new AppDbContext())
{
    symbols = db.Stocks.Select(s => s.Symbol).ToList();
}

foreach (var symbol in symbols)
{
    try
    {
        await GetCandles(symbol);
        await Predict(symbol);
        await AnalyzePredicitons(symbol);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error fetching candles for {symbol}: {ex.Message}");
    }
}

Console.WriteLine("Done.");
Console.ReadLine();
