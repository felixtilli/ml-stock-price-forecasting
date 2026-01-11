using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using MLStockPriceForecasting.Models;

public class AppDbContext : DbContext
{
    public DbSet<Stock> Stocks { get; set; }
    public DbSet<ForecastingStrategy> ForecastingStrategies { get; set; }
    public DbSet<Forecast> Forecasts { get; set; }
    public DbSet<Value> Values { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
    {
        var config = new ConfigurationBuilder().AddJsonFile("appsettings.json").Build();

        options.UseSqlServer(config.GetConnectionString("Database"));
    }
}
