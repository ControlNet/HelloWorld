// global trend section
function runGlobalTrend() {
    // init
    const svgWidth = +d3.select("#global-trend-left").attr("width").slice(0, -2);
    const svgHeight = +d3.select("#global-trend-left").attr("height").slice(0, -2);

    const margin = {top: 10, right: 150, bottom: 30, left: 45};
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    // select svg
    const svgLeft = d3.select("#global-trend-left");
    const svgRight = d3.select("#global-trend-right");

    // build g tags to perform margin
    const lineChartLeft = svgLeft
        .append("g")
        .classed("line-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const lineChartRight = svgRight
        .append("g")
        .classed("line-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // read csv file to make the visualizations
    d3.csv("data/covid19_timeseries.csv", data => {
        const dates = data.map(row => utils.parseDate(row.date));
        // choose colors to represent the category
        const colorMap = {
            confirmed: "#82AAFF", deaths: "#F07178", recovered: "#C3E88D", newcase: "#616161",
            infection_rate: "#F07178", recovery_rate: "#C3E88D"
        };

        // initialize the tooltip elements
        let hoveredDate;
        let xAxisLeft, xAxisRight, yAxisLeft, yAxisRight, dataObjsLeft, dataObjsRight;
        const {tooltip: tooltipLeft, tooltipRect: tooltipRectLeft, tooltipText: tooltipTextLeft} =
            utils.initTooltip(svgLeft, 5, 130);
        const {tooltip: tooltipRight, tooltipRect: tooltipRectRight, tooltipText: tooltipTextRight} =
            utils.initTooltip(svgRight, 3, 130);

        // a function for updating tooltips
        function updateTooltip(pos, mousePos) {
            // remove all old circles
            d3.select("#global-trend-section .svg-container")
                .selectAll("g.circles")
                .remove()

            // update the location of vertical lines
            const xLeft = xAxisLeft(hoveredDate);
            const xRight = xAxisRight(hoveredDate);

            d3.select("#global-trend-section .tooltip-line-left")
                .attr("x1", xLeft)
                .attr("x2", xLeft)
                .style("opacity", 1);

            d3.select("#global-trend-section .tooltip-line-right")
                .attr("x1", xRight)
                .attr("x2", xRight)
                .style("opacity", 1);

            const dataObjsLeftSelected = utils.selectDataByDate(dataObjsLeft, hoveredDate);
            const dataObjsRightSelected = utils.selectDataByDate(dataObjsRight, hoveredDate);

            // highlight the data point in the vertical lines
            utils.addTooltipCircles(lineChartLeft, colorMap, dataObjsLeft, dataObjsLeftSelected, xLeft, yAxisLeft);
            utils.addTooltipCircles(lineChartRight, colorMap, dataObjsRight, dataObjsRightSelected, xRight, yAxisRight);

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
                        `Recovered: ${dataObjsLeftSelected.recovered}`,
                        `Deaths: ${dataObjsLeftSelected.deaths}`,
                        `New Case: ${dataObjsLeftSelected.newcase}`
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
                        `Infection Rate: ${Math.round(dataObjsRightSelected.infection_rate * 1000) / 1000}`,
                        `Recovery Rate: ${Math.round(dataObjsRightSelected.recovery_rate * 1000) / 1000}`
                    ], mousePos)
                    break;
            }
        }

        // the event for updating the tooltip
        function updateTooltipEvent(xAxis, pos) {
            return function() {
                const mousePos = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
                const xValue = xAxis.invert(mousePos.x - margin.left);
                hoveredDate = utils.findClosest(dates, xValue);
                updateTooltip(pos, mousePos);
            };
        }

        // the function for initializing the left line chart
        function initLineChartLeft() {
            const categories = ["confirmed", "deaths", "recovered", "newcase"];
            // Reformat the data as an array of {x, y} tuples
            const dataObjsAll = utils.melt(categories, data);
            const dataObjsPath = dataObjsAll.filter(each => each.category !== "newcase");
            const dataObjsBins = dataObjsAll.filter(each => each.category === "newcase");
            // x axis
            const x = utils.buildXAxisWithDate(lineChartLeft, data, width, height);

            // y axis for lines
            const maxValue = Math.max(...data.map(row => +row["confirmed"]));
            const y = d3.scaleLinear()
                .domain([0, maxValue])
                .range([height, 0]);
            // y axis for new case bar chart
            const y2 = d3.scaleLinear()
                .domain([0, maxValue / 20])
                .range([height, 0]);
            // generate sticks
            lineChartLeft.append("g")
                .call(d3.axisLeft(y).tickFormat(d => d / 1e6));

            lineChartLeft.append("g")
                .attr("transform", `translate(${width}, 0)`)
                .call(d3.axisRight(y2).tickFormat(d => d / 1e6));

            // add y label
            lineChartLeft.append("text")
                .text("Number (Million)")
                .attr("transform", `translate(${-30}, ${height / 2}) rotate(270)`)
                .style("font-size", 12)
                .style("text-anchor", "middle");

            // reassign to an outer variable for other function access
            xAxisLeft = x;
            yAxisLeft = [y, y, y, y2];
            dataObjsLeft = dataObjsAll;

            // Add the lines
            const line = utils.addLine(x, y);
            // Add vertical lines for hovering
            utils.initTooltipLine("left", lineChartLeft, y, maxValue);

            lineChartLeft.append("g")
                .classed("trends", true)
                .classed("line-chart-element", true)
                .style("stroke-width", 3)
                .style("fill", "None")
                .selectAll()
                .data(dataObjsBins).enter()
                .append("path")
                .attr("stroke", "none")
                .attr("fill", d => colorMap[d.category])
                .attr("d", d => d3.area().x(d => x(d["date"])).y0(y2(0)).y1(d => y2(d["value"]))(d.values));

            // add label
            lineChartLeft.append("text")
                .text("New Case (Million)")
                .attr("transform", `translate(${width + 40}, ${height / 2}) rotate(270)`)
                .style("font-size", 12)
                .style("text-anchor", "middle")

            // add path for each category
            utils.addPath(lineChartLeft, dataObjsPath, line, colorMap, false);
            // set event for tooltip
            svgLeft.on("mousemove", updateTooltipEvent(x, "left"));
        }

        // the function for initializing the right line chart
        function initLineChartRight() {
            const categories = ["infection_rate", "recovery_rate"];
            // Reformat the data as an array of {x, y} tuples
            const dataObjs = utils.melt(categories, data);
            // x axis
            const x = utils.buildXAxisWithDate(lineChartRight, data, width, height);
            // y axis
            const maxValue = Math.max(...data.map(row => +row["recovery_rate"]));
            const y = d3.scaleLinear()
                .domain([0, Math.max(...data.map(row => +row["recovery_rate"]))])
                .range([height, 0]);
            // reassign to an outer variable for other function access
            xAxisRight = x;
            yAxisRight = [y, y];
            dataObjsRight = dataObjs;
            // generate sticks
            lineChartRight.append("g")
                .call(d3.axisLeft(y));
            // add y label
            lineChartRight.append("text")
                .text("Rate")
                .attr("transform", `translate(${-35}, ${height / 2}) rotate(270)`)
                .style("font-size", 12)
                .style("text-anchor", "middle")
            // Add the lines
            const line = utils.addLine(x, y);
            // Add vertical lines for hovering
            utils.initTooltipLine("right", lineChartRight, y, maxValue);
            // for each category
            utils.addPath(lineChartRight, dataObjs, line, colorMap);
            // set event for tooltip
            svgRight.on("mousemove", updateTooltipEvent(x, "right"));
        }

        // hide all tooltips (vertical lines, text rect, and circles) when the user do not focus on the section
        [svgLeft, svgRight].forEach(each => {
            each.on("mouseleave", function() {
                d3.selectAll("#global-trend-section .tooltip")
                    .style("opacity", 0);
            });
        });

        // add line charts
        initLineChartLeft();
        initLineChartRight();
        // add legends for both line charts
        utils.addLegendRight(lineChartLeft, width, ["confirmed", "deaths", "recovered", "newcase"], colorMap,
            ["Confirmed", "Deaths", "Recovered", "New Case"], true);
        utils.addLegendRight(lineChartRight, width, ["infection_rate", "recovery_rate"], colorMap,
            ["Infection Rate", "Recovery Rate"], false);
    });
}

runGlobalTrend()