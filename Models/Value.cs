using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace MLStockPriceForecasting.Models;

public class Value
{
    [Key]
    public long Id { get; set; }
    [ForeignKey("Stock")]
    public long StockId { get; set; }
    public DateTime Date { get; set; }
    public float Close { get; set; }
    public float High { get; set; }
    public float Low { get; set; }
    public float TradeCount { get; set; }
    public float Open { get; set; }
    public float Volume { get; set; }
    public float Vwap { get; set; }
    public virtual Stock Stock { get; set; }
}
