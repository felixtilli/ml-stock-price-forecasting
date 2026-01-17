using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace MLStockPriceForecasting.Models;

public class ForecastResult
{
    [Key]
    public long Id { get; set; }
    [ForeignKey("Forecast")]
    public long ForecastId { get; set; }
    public bool DirectionCorrect { get; set; }
    public float Diff { get; set; }
    public virtual Forecast Forecast { get; set; }
}
