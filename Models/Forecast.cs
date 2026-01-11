using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace MLStockPriceForecasting.Models;

public class Forecast
{
    [Key]
    public long Id { get; set; }
    [ForeignKey("Stock")]
    public long StockId { get; set; }
    public DateTime Date { get; set; }
    public float ClosePrice { get; set; }
    public virtual Stock Stock { get; set; }
    [ForeignKey("ForecastingStrategy")]
    public long ForecastingStrategyId { get; set; }
    public virtual ForecastingStrategy ForecastingStrategy { get; set; }
}
