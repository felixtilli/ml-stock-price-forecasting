using Alpaca.Markets;

namespace MLStockPriceForecasting.Models;

public class Candle
{
    public float Close { get; set; }
    public float High { get; set; }
    public float Low { get; set; }
    public float TradeCount { get; set; }
    public float Open { get; set; }
    public DateTime TimeUtc { get; set; }
    public float Volume { get; set; }
    public float Vwap { get; set; }

    public static Candle ConvertIBarToCandle(IBar item) => new Candle
    {
        Close = Convert.ToSingle(item.Close),
        High = Convert.ToSingle(item.High),
        Low = Convert.ToSingle(item.Low),
        Open = Convert.ToSingle(item.Open),
        Volume = Convert.ToSingle(item.Volume),
        Vwap = Convert.ToSingle(item.Vwap),
        TradeCount = Convert.ToSingle(item.TradeCount),
        TimeUtc = item.TimeUtc
    };
}
