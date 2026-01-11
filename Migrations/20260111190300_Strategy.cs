using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MLStockPriceForecasting.Migrations
{
    /// <inheritdoc />
    public partial class Strategy : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<long>(
                name: "ForecastingStrategyId",
                table: "Forecasts",
                type: "bigint",
                nullable: false,
                defaultValue: 0L);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ForecastingStrategyId",
                table: "Forecasts");
        }
    }
}
