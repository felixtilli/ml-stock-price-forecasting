using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MLStockPriceForecasting.Migrations
{
    /// <inheritdoc />
    public partial class ForecastResult : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "ForecastResults",
                columns: table => new
                {
                    Id = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    ForecastId = table.Column<long>(type: "bigint", nullable: false),
                    DirectionCorrect = table.Column<bool>(type: "bit", nullable: false),
                    Diff = table.Column<float>(type: "real", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ForecastResults", x => x.Id);
                    table.ForeignKey(
                        name: "FK_ForecastResults_Forecasts_ForecastId",
                        column: x => x.ForecastId,
                        principalTable: "Forecasts",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Forecasts_ForecastingStrategyId",
                table: "Forecasts",
                column: "ForecastingStrategyId");

            migrationBuilder.CreateIndex(
                name: "IX_ForecastResults_ForecastId",
                table: "ForecastResults",
                column: "ForecastId");

            migrationBuilder.AddForeignKey(
                name: "FK_Forecasts_ForecastingStrategies_ForecastingStrategyId",
                table: "Forecasts",
                column: "ForecastingStrategyId",
                principalTable: "ForecastingStrategies",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_Forecasts_ForecastingStrategies_ForecastingStrategyId",
                table: "Forecasts");

            migrationBuilder.DropTable(
                name: "ForecastResults");

            migrationBuilder.DropIndex(
                name: "IX_Forecasts_ForecastingStrategyId",
                table: "Forecasts");
        }
    }
}
