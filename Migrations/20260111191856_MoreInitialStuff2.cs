using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MLStockPriceForecasting.Migrations
{
    /// <inheritdoc />
    public partial class MoreInitialStuff2 : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "TimeUtc",
                table: "Values");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<DateTime>(
                name: "TimeUtc",
                table: "Values",
                type: "datetime2",
                nullable: false,
                defaultValue: new DateTime(1, 1, 1, 0, 0, 0, 0, DateTimeKind.Unspecified));
        }
    }
}
