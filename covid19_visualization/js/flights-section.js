// flights section
function runFlightsSection() {
    // init
    const svgWidth = +d3.select("#flights-trend-left").attr("width").slice(0, -2);
    const svgHeight = +d3.select("#flights-trend-left").attr("height").slice(0, -2);

    const margin = {top: 10, right: 150, bottom: 30, left: 45};
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    // svg selection
    const svgLeft = d3.select("#flights-trend-left");
    const svgRight = d3.select("#flights-trend-right");

    // append g tags to perform margin
    const lineChartLeft = svgLeft
        .append("g")
        .classed("line-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const lineChartRight = svgRight
        .append("g")
        .classed("line-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // read csv
    d3.csv("data/flights.csv", data => {
        const dates = data.map(row => utils.parseDate(row.date));
        // choose colors to represent the category
        const colorMap = {
            confirmed: "#82AAFF", total_flights: "#F78C6C", commercial_flights: "#C3E88D",
            commercial_rate: "#FFCB6B"
        };

        // init tooltip elements
        let hoveredDate;
        let xAxisLeft, xAxisRight, yAxisLeft, yAxisRight, dataObjsLeft, dataObjsRight;
        const {tooltip: tooltipLeft, tooltipRect: tooltipRectLeft, tooltipText: tooltipTextLeft} =
            utils.initTooltip(svgLeft, 4, 160);
        const {tooltip: tooltipRight, tooltipRect: tooltipRectRight, tooltipText: tooltipTextRight} =
            utils.initTooltip(svgRight, 3, 145);

        function updateTooltip(pos, mousePos) {
            // remove all old circles
            d3.select("#flights-section .svg-container")
                .selectAll("g.circles")
                .remove()

            // update the location of vertical lines
            const xLeft = xAxisLeft(hoveredDate);
            const xRight = xAxisRight(hoveredDate);

            d3.select("#flights-section .tooltip-line-left")
                .attr("x1", xLeft)
                .attr("x2", xLeft)
                .style("opacity", 1);

            d3.select("#flights-section .tooltip-line-right")
                .attr("x1", xRight)
                .attr("x2", xRight)
                .style("opacity", 1);

            const dataObjsLeftSelected = utils.selectDataByDate(dataObjsLeft, hoveredDate);
            const dataObjsRightSelected = utils.selectDataByDate(dataObjsRight, hoveredDate);

            // highlight the data point in the vertical lines
            utils.addTooltipCircles(lineChartLeft, colorMap, dataObjsLeft, dataObjsLeftSelected, xLeft, yAxisLeft);
            utils.addTooltipCircles(lineChartRight, colorMap, dataObjsRight, dataObjsRightSelected, xRight, yAxisRight);

            // add text for tooltip
            switch (pos) {
                case "left":
                    tooltipLeft.style("opacity", 1);
                    tooltipRight.style("opacity", 0);
                    tooltipRectLeft
                        .attr("x", (mousePos.x + 10) + "px")
                        .attr("y", (mousePos.y - 55) + "px");
                    utils.setTooltipText(tooltipTextLeft, [
                        `${hoveredDate.toISOString().slice(0, 10)}`,
                        `Confirmed: ${dataObjsLeftSelected.confirmed}`,
                        `Total Flights: ${dataObjsLeftSelected.total_flights}`,
                        `Commercial Flights: ${dataObjsLeftSelected.commercial_flights}`
                    ], mousePos)
                    break;
                case "right":
                    tooltipLeft.style("opacity", 0);
                    tooltipRight.style("opacity", 1);
                    tooltipRectRight
                        .attr("x", (mousePos.x + 10) + "px")
                        .attr("y", (mousePos.y - 55) + "px");
                    utils.setTooltipText(tooltipTextRight, [
                        `${hoveredDate.toISOString().slice(0, 10)}`,
                        `Confirmed: ${dataObjsRightSelected.confirmed}`,
                        `Commercial Rate: ${Math.round(dataObjsRightSelected.commercial_rate * 1000) / 1000}`
                    ], mousePos)
                    break;
            }
        }

        // the event for updating the tooltips
        function updateTooltipEvent(xAxis, pos) {
            return function() {
                const mousePos = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
                const xValue = xAxis.invert(mousePos.x - margin.left);
                hoveredDate = utils.findClosest(dates, xValue);
                updateTooltip(pos, mousePos);
            };
        }

        // init the left line chart
        function initLineChartLeft() {
            const categories = ["confirmed", "total_flights", "commercial_flights"];
            // Reformat the data as an array of {x, y} tuples
            const dataObjs = utils.melt(categories, data);
            const dataObjsLeftAxis = dataObjs.filter(each => each.category === "confirmed");
            const dataObjsRightAxis = dataObjs.filter(each => each.category !== "confirmed");

            // x axis
            const x = utils.buildXAxisWithDate(lineChartLeft, data, width, height);
            // y axis for lines
            const maxValue = Math.max(...data.map(row => +row["confirmed"]));
            const y = d3.scaleLinear()
                .domain([0, maxValue])
                .range([height, 0]);
            // y axis for new case bar chart
            const y2 = d3.scaleLinear()
                .domain([0, maxValue / 100])
                .range([height, 0]);
            // generate sticks
            lineChartLeft.append("g")
                .call(d3.axisLeft(y).tickFormat(d => d / 1e6));

            lineChartLeft.append("g")
                .attr("transform", `translate(${width}, 0)`)
                .call(d3.axisRight(y2).tickFormat(d => d / 1e3));

            // add y label
            lineChartLeft.append("text")
                .text("Confirmed (Million)")
                .attr("transform", `translate(${-30}, ${height / 2}) rotate(270)`)
                .style("font-size", 12)
                .style("text-anchor", "middle");

            // reassign to an outer variable for other function access
            xAxisLeft = x;
            yAxisLeft = [y, y2, y2];
            dataObjsLeft = dataObjs;

            // Add vertical lines for hovering
            utils.initTooltipLine("left", lineChartLeft, y, maxValue);
            // Add the lines
            const line = utils.addLine(x, y);
            const line2 = utils.addLine(x, y2);
            // add path for each category
            utils.addPath(lineChartLeft, dataObjsLeftAxis, line, colorMap);
            utils.addPath(lineChartLeft, dataObjsRightAxis, line2, colorMap);

            // add label
            lineChartLeft.append("text")
                .text("Flights (Thousands)")
                .attr("transform", `translate(${width + 40}, ${height / 2}) rotate(270)`)
                .style("font-size", 12)
                .style("text-anchor", "middle")

            // set event for tooltip
            svgLeft.on("mousemove", updateTooltipEvent(x, "left"));
        }

        // init the right line chart
        function initLineChartRight() {
            const categories = ["confirmed", "commercial_rate"];
            // Reformat the data as an array of {x, y} tuples
            const dataObjs = utils.melt(categories, data);
            const dataObjsLeftAxis = dataObjs.filter(each => each.category === "confirmed");
            const dataObjsRightAxis = dataObjs.filter(each => each.category !== "confirmed");
            // x axis
            const x = utils.buildXAxisWithDate(lineChartRight, data, width, height);
            // y axis
            const maxValue = Math.max(...data.map(row => +row["confirmed"]));

            const y = d3.scaleLinear()
                .domain([0, maxValue])
                .range([height, 0]);
            // y axis for new case bar chart
            const y2 = d3.scaleLinear()
                .domain([0, 1])
                .range([height, 0]);
            // generate sticks
            lineChartRight.append("g")
                .call(d3.axisLeft(y).tickFormat(d => d / 1e6));

            lineChartRight.append("g")
                .attr("transform", `translate(${width}, 0)`)
                .call(d3.axisRight(y2).tickFormat(d => d));

            // reassign to an outer variable for other function access
            xAxisRight = x;
            yAxisRight = [y, y2];
            dataObjsRight = dataObjs;

            // add y label
            lineChartRight.append("text")
                .text("Confirmed (Million)")
                .attr("transform", `translate(${-30}, ${height / 2}) rotate(270)`)
                .style("font-size", 12)
                .style("text-anchor", "middle");

            // Add the lines
            const line = utils.addLine(x, y);
            const line2 = utils.addLine(x, y2);
            // Add vertical lines for hovering
            utils.initTooltipLine("right", lineChartRight, y, maxValue);
            // for each category
            utils.addPath(lineChartRight, dataObjsLeftAxis, line, colorMap);
            utils.addPath(lineChartRight, dataObjsRightAxis, line2, colorMap);
            // add label
            lineChartRight.append("text")
                .text("Commercial Rate")
                .attr("transform", `translate(${width + 40}, ${height / 2}) rotate(270)`)
                .style("font-size", 12)
                .style("text-anchor", "middle")
            // set event for tooltip
            svgRight.on("mousemove", updateTooltipEvent(x, "right"));
        }

        // hide all tooltips (vertical lines, text rect, and circles) when the user do not focus on the section
        [svgLeft, svgRight].forEach(each => {
            each.on("mouseleave", function() {
                d3.selectAll("#flights-section .tooltip")
                    .style("opacity", 0);
            });
        });

        // build line charts
        initLineChartLeft();
        initLineChartRight();
        // add legend
        utils.addLegendRight(lineChartLeft, width, ["confirmed", "total_flights", "commercial_flights"], colorMap,
            ["Confirmed", "Total Flights", "Comm. Flights"], true);
        utils.addLegendRight(lineChartRight, width, ["confirmed", "commercial_rate"], colorMap,
            ["Confirmed", "Comm. Rate"], true);
    });
}

runFlightsSection()