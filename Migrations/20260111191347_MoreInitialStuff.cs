using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MLStockPriceForecasting.Migrations
{
    /// <inheritdoc />
    public partial class MoreInitialStuff : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "ClosePrice",
                table: "Values",
                newName: "Vwap");

            migrationBuilder.AddColumn<float>(
                name: "Close",
                table: "Values",
                type: "real",
                nullable: false,
                defaultValue: 0f);

            migrationBuilder.AddColumn<float>(
                name: "High",
                table: "Values",
                type: "real",
                nullable: false,
                defaultValue: 0f);

            migrationBuilder.AddColumn<float>(
                name: "Low",
                table: "Values",
                type: "real",
                nullable: false,
                defaultValue: 0f);

            migrationBuilder.AddColumn<float>(
                name: "Open",
                table: "Values",
                type: "real",
                nullable: false,
                defaultValue: 0f);

            migrationBuilder.AddColumn<DateTime>(
                name: "TimeUtc",
                table: "Values",
                type: "datetime2",
                nullable: false,
                defaultValue: new DateTime(1, 1, 1, 0, 0, 0, 0, DateTimeKind.Unspecified));

            migrationBuilder.AddColumn<float>(
                name: "TradeCount",
                table: "Values",
                type: "real",
                nullable: false,
                defaultValue: 0f);

            migrationBuilder.AddColumn<float>(
                name: "Volume",
                table: "Values",
                type: "real",
                nullable: false,
                defaultValue: 0f);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Close",
                table: "Values");

            migrationBuilder.DropColumn(
                name: "High",
                table: "Values");

            migrationBuilder.DropColumn(
                name: "Low",
                table: "Values");

            migrationBuilder.DropColumn(
                name: "Open",
                table: "Values");

            migrationBuilder.DropColumn(
                name: "TimeUtc",
                table: "Values");

            migrationBuilder.DropColumn(
                name: "TradeCount",
                table: "Values");

            migrationBuilder.DropColumn(
                name: "Volume",
                table: "Values");

            migrationBuilder.RenameColumn(
                name: "Vwap",
                table: "Values",
                newName: "ClosePrice");
        }
    }
}
