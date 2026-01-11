namespace MLStockPriceForecasting.Models;

public class FeatureCandle
{
    public float Close { get; set; }
    public float CloseLag1 { get; set; }
    public float CloseLag2 { get; set; }
    public float CloseLag3 { get; set; }
}
