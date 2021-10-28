// This file is just a duplicate of nation-comparison-section.js.js in the case I do not understand the internal logic but I do need the same section twice

function runNationSection() {
    // init comparison plots
    const svgWidth = 530;
    const svgHeight = 300;

    const margin = {top: 10, right: 150, bottom: 30, left: 45};
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    const svgDetailLeft = d3.select("#nation-detail-left");
    const svgDetailRight = d3.select("#nation-detail-right");

    const detailLineChartLeft = svgDetailLeft
        .append("g")
        .classed("line-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const detailLineChartRight = svgDetailRight
        .append("g")
        .classed("line-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // init world map
    const projection = d3.geoMercator().scale(100).translate([320, 180]);
    const path = d3.geoPath().projection(projection);
    const svgMap = d3.select("#nation-detail-world-map");
    const svgColorBar = d3.select("#nation-detail-color-bar");
    const colorBar = svgColorBar.append("g")
        .attr("id", "nation-detail-color-bar-g")
        .classed("color-bar", true)
        .attr("transform", "translate(0, 10)");
    const descriptionContainer = d3.select("#nation-detail-description-container");

    // selected object of world map
    let selected = null;

    function updateColorBar(colorBar, colorScale, valueMin, valueMax) {
        const scale = document.getElementById("nation-world-map-scale2").value;
        const fill = document.getElementById("nation-world-map-fill2").value;
        utils.addColorBar(colorBar, colorScale, valueMin, valueMax, scale, fill, 360);
    }

    // curried function to build color map
    function fill2color(fill, scale, geoJson) {
        // extract values for fill color
        const values = geoJson.features.map(x => +x.properties[fill]);
        // min and max value of the array
        const maxValue = Math.max(...values)
        const minValue = Math.min(...values.filter(utils.isNotZero))
        // calculate the sorted orders
        const orders = utils.argsort(values);
        let value2range;
        let colorBarMax, colorBarMin;
        // get value2range function for color map
        switch (scale) {
            case "linear":
                colorBarMax = maxValue;
                colorBarMin = minValue;
                if (fill === "confirmed") {
                    value2range = d3.scaleLinear()
                        .domain([colorBarMin, colorBarMax])
                        .range([1, -4]);
                } else if (fill === "confirmed_per_capita") {
                    value2range = d3.scaleLinear()
                        .domain([colorBarMin, colorBarMax])
                        .range([1, -8]);
                }
                break;
            case "log":
                colorBarMax = maxValue;
                colorBarMin = minValue;
                value2range = d3.scaleLog()
                    .domain([colorBarMin, colorBarMax])
                    .range([1.2, -0.1]);
                break;
            case "percentile":
                colorBarMax = 100;
                colorBarMin = 0;
                value2range = d3.scaleLinear()
                    .domain([colorBarMin, colorBarMax])
                    .range([1.4, 0]);
                break;
        }

        updateColorBar(colorBar, x => d3.interpolateRdYlGn(value2range(x)),
            colorBarMin, colorBarMax);

        // if the value is 0, use "gray" color, else use the color generated
        return (i, value) => {
            if (value === 0) {
                return "gray";
            } else if (scale === "percentile") {
                return d3.interpolateRdYlGn(value2range(orders[i] / orders.length * 100));
            } else {
                return d3.interpolateRdYlGn(value2range(value));
            }
        };
    }

    let mapColor;

    function initMap() {
        const fill = document.getElementById("nation-world-map-fill2").value;
        const scale = document.getElementById("nation-world-map-scale2").value;

        d3.json("data/geojson.json", geoJson => {
            // build the color map for selected fill value
            const colorMap = fill2color(fill, scale, geoJson);
            mapColor = colorMap;
            // build paths with world-map-nation class
            const paths = svgMap.selectAll("path")
                .data(geoJson.features).enter()
                .append("path")
                .classed("world-map-nation", true)
                .attr("d", path);
            let selectedPath = null;
            // assign the interactivity for selected
            paths.style("fill", (row, i) => colorMap(i, row.properties[fill]))
                .on("mouseover", function() {
                    const element = d3.select(this);
                    element.style("fill", "#89DDFF");
                })
                .on("mouseout", function(row, i) {
                    const element = d3.select(this);
                    element.style("fill", colorMap(i, row.properties[fill]));
                })
                .on("click", function(row) {
                    // display nation name
                    d3.select("#nation-world-map-bottom-text")
                        .html(row.properties["nation"]);
                    // mark the selected nation
                    selected = row.properties;
                    if (utils.isNotNull(selectedPath)) {
                        d3.select(selectedPath).classed("selected", false);
                    }
                    d3.select(this).classed("selected", true);
                    selectedPath = this;

                    updateDescription();
                    updateNationDetailPlots();
                });
        })
    }

    async function updateMap() {
        // collect the parameter of selection
        const fill = document.getElementById("nation-world-map-fill2").value;
        const scale = document.getElementById("nation-world-map-scale2").value;

        await new Promise((resolve) => {
            d3.json("data/geojson.json", geoJson => {
                // build the color map for selected fill value
                const colorMap = fill2color(fill, scale, geoJson);
                resolve(true);
                mapColor = colorMap;
                const paths = svgMap.selectAll("path");

                // assign the interactivity for selected
                paths.transition()
                    .style("fill", (row, i) => colorMap(i, row.properties[fill]));

                paths.on("mouseout", function(row, i) {
                    const element = d3.select(this);
                    element.style("fill", colorMap(i, row.properties[fill]));
                });
            });
        })
    }

    let previousSelected = null;

    function updateNationDetailPlots() {
        if (selected === previousSelected) {
            // if not change, skip the update
            return;
        }
        // remove previous
        const oldComponentsLeft = [
            d3.select("#nation-detail-left g.line-chart").selectAll("*"),
            d3.select("#nation-detail-left g.tooltip").selectAll("*")
        ]

        const oldComponentsRight = [
            d3.select("#nation-detail-right g.line-chart").selectAll("*"),
            d3.select("#nation-detail-right g.tooltip").selectAll("*")
        ]

        if (utils.isNotNull(selected) && previousSelected === null) {
            // if firstly select the nation
            d3.select("#nation-detail-title").html("");
            svgDetailLeft.attr("height", svgHeight);
            svgDetailRight.attr("height", svgHeight);
        } else if (selected === null) {
            // unselect the nation, remove all and display the hint
            d3.select("#nation-detail-title").html("Please Select Nation in the Map");
            svgDetailLeft.attr("height", "0px");
            svgDetailRight.attr("height", "0px");
            oldComponentsLeft.forEach(each => each.remove());
            oldComponentsRight.forEach(each => each.remove());
            previousSelected = null;
            return;
        }

        previousSelected = selected;

        const nation = selected["nation"];

        d3.csv("data/covid19_nations_confirmed.csv", confirmedData => {
            d3.csv("data/covid19_nations_deaths.csv", deathsData => {
                d3.csv("data/covid19_nations_recovered.csv", recoveredData => {
                    d3.csv("data/covid19_nations_newcase.csv", newcaseData => {
                        update(confirmedData, deathsData, recoveredData, newcaseData);
                    })
                })
            })
        })

        function update(confirmedData, deathsData, recoveredData, newcaseData) {
            const dates = confirmedData.map(row => utils.parseDate(row.date));
            // choose colors to represent the category
            const detailPlotsColorMap = {
                confirmed: "#82AAFF", deaths: "#F07178", recovered: "#C3E88D", newcase: "#616161",
                infection_rate: "#F07178", recovery_rate: "#C3E88D"
            };

            const data = utils.range(dates.length)
                .map(index => {
                    return {
                        date: confirmedData.map(row => row["date"])[index],
                        confirmed: +confirmedData.map(row => row[nation])[index],
                        deaths: +deathsData.map(row => row[nation])[index],
                        recovered: +recoveredData.map(row => row[nation])[index],
                        newcase: +newcaseData.map(row => row[nation])[index]
                    };
                }).map(row => {
                    if (row["confirmed"] === 0) {
                        row["infection_rate"] = 0;
                        row["recovery_rate"] = 0;
                    } else {
                        row["infection_rate"] = row["newcase"] / row["confirmed"];
                        row["recovery_rate"] = row["recovered"] / row["confirmed"];
                    }
                    return row;
                });

            let hoveredDate;
            let xAxisLeft, xAxisRight, yAxisLeft, yAxisRight, dataObjsLeft, dataObjsRight;
            const {tooltip: tooltipLeft, tooltipRect: tooltipRectLeft, tooltipText: tooltipTextLeft} =
                utils.initTooltip(svgDetailLeft, 5, 130);
            const {tooltip: tooltipRight, tooltipRect: tooltipRectRight, tooltipText: tooltipTextRight} =
                utils.initTooltip(svgDetailRight, 3, 130);

            function updateTooltip(pos, mousePos) {
                // remove all old circles
                d3.select("#nation-detail-div")
                    .selectAll("g.circles")
                    .remove()

                // update the location of vertical lines
                const xLeft = xAxisLeft(hoveredDate);
                const xRight = xAxisRight(hoveredDate);

                d3.select("#nation-detail-div .tooltip-line-left")
                    .attr("x1", xLeft)
                    .attr("x2", xLeft)
                    .style("opacity", 1);

                d3.select("#nation-detail-div .tooltip-line-right")
                    .attr("x1", xRight)
                    .attr("x2", xRight)
                    .style("opacity", 1);

                const dataObjsLeftSelected = utils.selectDataByDate(dataObjsLeft, hoveredDate);
                const dataObjsRightSelected = utils.selectDataByDate(dataObjsRight, hoveredDate);

                // highlight the data point in the vertical lines
                utils.addTooltipCircles(detailLineChartLeft, detailPlotsColorMap, dataObjsLeft, dataObjsLeftSelected,
                    xLeft, yAxisLeft);
                utils.addTooltipCircles(detailLineChartRight, detailPlotsColorMap, dataObjsRight, dataObjsRightSelected,
                    xRight, yAxisRight);

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

            function updateTooltipEvent(xAxis, pos) {
                return function() {
                    const mousePos = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
                    const xValue = xAxis.invert(mousePos.x - margin.left);
                    hoveredDate = utils.findClosest(dates, xValue);
                    updateTooltip(pos, mousePos);
                };
            }

            function initLineChartLeft() {
                const categories = ["confirmed", "deaths", "recovered", "newcase"];
                // Reformat the data as an array of {x, y} tuples
                const dataObjsAll = utils.melt(categories, data);
                const dataObjsPath = dataObjsAll.filter(each => each.category !== "newcase");
                const dataObjsBins = dataObjsAll.filter(each => each.category === "newcase");
                // x axis
                const x = utils.buildXAxisWithDate(detailLineChartLeft, data, width, height);

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
                detailLineChartLeft.append("g")
                    .call(d3.axisLeft(y).tickFormat(d => d / 1e6));

                detailLineChartLeft.append("g")
                    .attr("transform", `translate(${width}, 0)`)
                    .call(d3.axisRight(y2).tickFormat(d => d / 1e6));

                // add y label
                detailLineChartLeft.append("text")
                    .text("Number (Million)")
                    .attr("transform", `translate(${-30}, ${height / 2}) rotate(270)`)
                    .style("font-size", 14)
                    .style("text-anchor", "middle");

                // reassign to an outer variable for other function access
                xAxisLeft = x;
                yAxisLeft = [y, y, y, y2];
                dataObjsLeft = dataObjsAll;

                // Add the lines
                const line = utils.addLine(x, y);
                // Add vertical lines for hovering
                utils.initTooltipLine("left", detailLineChartLeft, y, maxValue);

                detailLineChartLeft.append("g")
                    .classed("trends", true)
                    .classed("line-chart-element", true)
                    .style("stroke-width", 3)
                    .style("fill", "None")
                    .selectAll()
                    .data(dataObjsBins).enter()
                    .append("path")
                    .attr("stroke", "none")
                    .attr("fill", d => detailPlotsColorMap[d.category])
                    .attr("d", d => d3.area()
                        .x(d => x(d["date"]))
                        .y0(y2(0))
                        .y1(d => y2(d["value"]))(d.values));

                // add label
                detailLineChartLeft.append("text")
                    .text("New Case (Million)")
                    .attr("transform", `translate(${width + 40}, ${height / 2}) rotate(270)`)
                    .style("font-size", 14)
                    .style("text-anchor", "middle")

                // remove old components
                oldComponentsLeft.forEach(each => each.remove());
                // add path for each category
                utils.addPath(detailLineChartLeft, dataObjsPath, line, detailPlotsColorMap, false);
                // set event for tooltip
                svgDetailLeft.on("mousemove", updateTooltipEvent(x, "left"));
            }

            function initLineChartRight() {
                const categories = ["infection_rate", "recovery_rate"];
                // Reformat the data as an array of {x, y} tuples
                const dataObjs = utils.melt(categories, data);
                // x axis
                const x = utils.buildXAxisWithDate(detailLineChartRight, data, width, height);
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
                detailLineChartRight.append("g")
                    .call(d3.axisLeft(y));
                // add y label
                detailLineChartRight.append("text")
                    .text("Rate")
                    .attr("transform", `translate(${-35}, ${height / 2}) rotate(270)`)
                    .style("font-size", 14)
                    .style("text-anchor", "middle")
                // Add the lines
                const line = utils.addLine(x, y);
                // Add vertical lines for hovering
                utils.initTooltipLine("right", detailLineChartRight, y, maxValue);
                // remove old components
                oldComponentsRight.forEach(each => each.remove());
                // for each category
                utils.addPath(detailLineChartRight, dataObjs, line, detailPlotsColorMap);
                // set event for tooltip
                svgDetailRight.on("mousemove", updateTooltipEvent(x, "right"));
            }

            // hide all tooltips (vertical lines, text rect, and circles) when the user do not focus on the section
            [svgDetailLeft, svgDetailRight].forEach(each => {
                each.on("mouseleave", function() {
                    d3.selectAll("#nation-detail-div .tooltip")
                        .style("opacity", 0);
                });
            });

            initLineChartLeft();
            initLineChartRight();
            utils.addLegendRight(detailLineChartLeft, width, ["confirmed", "deaths", "recovered", "newcase"],
                detailPlotsColorMap, ["Confirmed", "Deaths", "Recovered", "New Case"], true);
            utils.addLegendRight(detailLineChartRight, width, ["infection_rate", "recovery_rate"],
                detailPlotsColorMap, ["Infection Rate", "Recovery Rate"], false);
        }
    }

    function updateDescription() {
        if (utils.isNotNull(selected)) {
            const fill = document.getElementById("nation-world-map-fill2").value;
            const value = selected[fill];
            let color;

            const paths = d3.select("#nation-detail-world-map")
                .selectAll("path")
                .each((d, i) => {
                    if (d.properties["nation"] === selected["nation"]) {
                        color = mapColor(i, value);
                    }
                })

            descriptionContainer
                .transition()
                .style("border-color", color);

            descriptionContainer.html(`<p style="color:${color}; font-size:20pt">${selected.nation}</p>
<p>Confirmed:</p>
<p style="text-align: right">${selected["confirmed"]}</p>
<p>Confirmed Per Millions:</p>
<p style="text-align: right">${Math.round(selected["confirmed_per_capita"] * 1e6)}</p>
<p>Deaths:</p>
<p style="text-align: right">${selected["deaths"]}</p>
<p>Recovered:</p>
<p style="text-align: right">${selected["recovered"]}</p>
`)
        }
    }

    initMap();

    const update = () => {
        updateMap().then(() => {
            updateDescription();
        })
    }
    document.getElementById("nation-world-map-fill2").onchange = update;
    document.getElementById("nation-world-map-scale2").onchange = update;
}

runNationSection()