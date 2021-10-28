function runNationSection() {
    // init comparison plots
    const svgWidth = +d3.select("#comparison-left").attr("width").slice(0, -2);
    const svgHeight = +d3.select("#comparison-left").attr("height").slice(0, -2);

    const margin = {top: 20, right: 20, bottom: 40, left: 45};
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    // svg selections
    const svgLeft = d3.select("#comparison-left");
    const svgMid = d3.select("#comparison-mid")
    const svgRight = d3.select("#comparison-right");
    const svgLegend = d3.select("#comparison-legend");
    const svgLegend2 = d3.select("#comparison-legend2")
    const svgColorBar = d3.select("#nation-comparison-color-bar");

    // add g tags for the svg elements
    const colorBar = svgColorBar.append("g")
        .attr("id", "nation-comparison-color-bar-g")
        .classed("color-bar", true)
        .attr("transform", "translate(0, 10)");

    const chartLeft = svgLeft
        .append("g")
        .classed("area-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const chartMid = svgMid
        .append("g")
        .classed("line-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const chartRight = svgRight
        .append("g")
        .classed("pie-chart", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const pieChartRight = chartRight
        .append("g")
        .classed("pie-chart-wrapper", true)
        .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

    // init world map
    const projection = d3.geoMercator().scale(90).translate([280, 145]);
    const path = d3.geoPath().projection(projection);
    const svgMap = d3.select("#nation-comparison-section .world-map");

    // color pool for new nations
    const colorPool = {
        values: ["#82AAFF", "#F07178", "#C3E88D", "#C792EA",
            "#467CDA", "#FFCB6B", "#F78C6C", "#89DDFF"],
        i: 0, // current color index

        // a method for generate new color from pool
        get: function(nation) {
            // if already in the color list
            if (utils.isNotNull(colorMap[nation])) {
                return colorMap[nation];
            } else {
                // else assign a new color to the nation
                const out = this.values[this.i];
                this.i++;
                if (this.i === this.values.length) {
                    this.i = 0;
                }
                return out;
            }
        }
    }

    // list the default nations in alphabet order
    const nationColors = {
        "Australia": "#C3E88D",
        "Brazil": "#FFCB6B",
        "China": "#F07178",
        "India": "#82AAFF",
        "Italy": "#C792EA",
        "Spain": "#F78C6C",
        "US": "#467CDA",
        "United Kingdom": "#89DDFF"
    };

    // main nation list
    const mainNations = Object.keys(nationColors);

    // main nation selection flags
    const mainNationsSelection = {
        "Brazil": true,
        "Australia": true,
        "China": true,
        "India": true,
        "Italy": true,
        "Spain": true,
        "US": true,
        "United Kingdom": true
    }

    // others color
    const othersColor = "#616161";

    // color map for default nations
    const colorMap = {
        "Australia": "#C3E88D",
        "Brazil": "#FFCB6B",
        "China": "#F07178",
        "India": "#82AAFF",
        "Italy": "#C792EA",
        "Spain": "#F78C6C",
        "US": "#467CDA",
        "United Kingdom": "#89DDFF",
        "Others": othersColor
    }

    // for other nations not in `mainNations`
    let selectedNations = utils.deepCopy(mainNations);

    // a boolean shows if the "Others" is selected
    let selectOthers = true;

    // selected object of world map
    let selected = null;

    // update color bar for this section
    function updateColorBar(colorBar, colorScale, valueMin, valueMax) {
        const scale = document.getElementById("nation-world-map-scale1").value;
        const fill = document.getElementById("nation-world-map-fill1").value;
        utils.addColorBar(colorBar, colorScale, valueMin, valueMax, scale, fill, 250);
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

    function initMap() {
        const fill = document.getElementById("nation-world-map-fill1").value;
        const scale = document.getElementById("nation-world-map-scale1").value;

        d3.json("data/geojson.json", geoJson => {
            // build the color map for selected fill value
            const colorMap = fill2color(fill, scale, geoJson);
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
                    d3.select("#comparison-world-map-bottom-text")
                        .html(row.properties["nation"]);
                    // mark the selected nation
                    selected = row.properties;
                    if (utils.isNotNull(selectedPath)) {
                        d3.select(selectedPath).classed("selected", false);
                    }
                    d3.select(this).classed("selected", true);
                    selectedPath = this;
                    if (!mainNations.includes(row.properties["nation"])) {
                        updateLegendSecondRow();
                        updateComparisonPlots();
                    }
                });
        })
    }

    function updateMap() {
        // collect the parameter of selection
        const fill = document.getElementById("nation-world-map-fill1").value;
        const scale = document.getElementById("nation-world-map-scale1").value;

        d3.json("data/geojson.json", geoJson => {
            // build the color map for selected fill value
            const colorMap = fill2color(fill, scale, geoJson);
            const paths = svgMap.selectAll("path");

            // assign the interactivity for selected
            paths.transition()
                .style("fill", (row, i) => colorMap(i, row.properties[fill]));

            paths.on("mouseout", function(row, i) {
                const element = d3.select(this);
                element.style("fill", colorMap(i, row.properties[fill]));
            });
        });
    }

    const customNationsLegendKeys = []

    function updateLegendSecondRow(removeIndex = -1) {
        if (removeIndex === -1) {
            // if the nation is already in the first row of legend, skip
            if (selected === null
                || mainNations.includes(selected["nation"])
                || customNationsLegendKeys.includes(selected["nation"])) {
                return;
            }
            // assign a new color for that
            colorMap[selected["nation"]] = colorPool.get(selected["nation"]);
            customNationsLegendKeys.push(selected["nation"]);
            // remove old
            svgLegend2.selectAll("g.legend").remove();
            // add legend in second row
            utils.addLegendBottom(svgLegend2, customNationsLegendKeys, colorMap,
                customNationsLegendKeys, false);
        } else {
            function moveLeft() {
                const obj = d3.select(this);
                const x = obj.attr("x");
                obj.transition().duration(500).attr("x", x - 85);
            }

            d3.selectAll(svgLegend2.select("g.legend").selectAll("rect, text").nodes().slice(removeIndex * 2))
                .each(moveLeft);

            setTimeout(updateLegendSecondRow, 510);
        }

        // add events
        svgLegend2.select("g.legend")
            .selectAll("rect")
            .each(function(d, i) {
                const rect = d3.select(this);
                const color = colorMap[d];
                const colorLight = d3.interpolate(color, "#FFF")(0.6);

                rect.on("mouseover", function() {
                    if (!legendIsFrozen) {
                        rect.style("fill", colorLight);
                    }
                }).on("mouseout", function() {
                    if (!legendIsFrozen) {
                        rect.style("fill", color);
                    }
                }).on("click", function() {
                    if (!legendIsFrozen) {
                        // remove the legend symbol
                        rect.remove();
                        d3.select(svgLegend2.select("g.legend").selectAll("text").nodes()[i])
                            .remove();

                        utils.removeItem(customNationsLegendKeys, d);
                        if (utils.isNotNull(selected) && selected["nation"] === d) {
                            selected = null;
                        }
                        svgMap.select("path.selected").classed("selected", false);
                        updateComparisonPlots();
                        updateLegendSecondRow(i);
                    }
                })
            })
    }

    function addPieChartAnnotation(arcAnnotation, d) {
        if (utils.isNotNull(d)) {
            // add annotation
            const [x, y] = arcAnnotation.centroid(d);
            const angle = (d["startAngle"] + d["endAngle"]) / 2;
            const connectorLength = 23;
            const dx = connectorLength * Math.sin(angle);
            const dy = -connectorLength * Math.cos(angle);

            pieChartRight.append("g")
                .classed("annotation-group", true)
                .call(utils.makeAnnotations(
                    [new utils.Annotation(
                        "", d.data["nation"], x, y, dx, dy, colorMap[d.data["nation"]], 50
                    )]
                ));
        }
    }

    let legendIsFrozen = false;
    let nationDataLatest;
    let sumValue;

    function freezeLegend(milliseconds) {
        legendIsFrozen = true;
        setTimeout(() => {
            legendIsFrozen = false
        }, milliseconds)
    }

    const pie = d3.pie()
        .value(d => d["value"])
        .sort(null);

    // set radius for pie chart
    const radius = Math.min(width, height) / 2
    // set arc functions for pie chart
    const arc = d3.arc()
        .innerRadius(radius - 80)
        .outerRadius(radius - 50);
    // the arc function for focused component
    const arcFocus = d3.arc()
        .innerRadius(radius - 70)
        .outerRadius(radius - 30);
    // the arc function for annotation
    const arcAnnotation = d3.arc()
        .innerRadius(radius - 50)
        .outerRadius(radius - 50);

    // init tooltip elements
    const {tooltip: tooltipLine, tooltipRect: tooltipRectLine, tooltipText: tooltipTextLine} =
        utils.initTooltip(svgMid, 3, 125);
    const {tooltip: tooltipArea, tooltipRect: tooltipRectArea, tooltipText: tooltipTextArea} =
        utils.initTooltip(svgLeft, 1, 100);
    const {tooltip: tooltipPie, tooltipRect: tooltipRectPie, tooltipText: tooltipTextPie} =
        utils.initTooltip(svgRight, 2, 125);

    // update tooltip for the line chart
    function updateTooltipForLine(hoveredData, hoveredPoint) {
        if (utils.isNotNull(hoveredData)) {
            const point = d3.select(hoveredPoint);
            const x = +point.attr("cx");
            const y = +point.attr("cy");

            // adjust the location of tooltip
            const tooltipX = (x + 65) > width ? (x - 100) : x;
            const tooltipY = (y - 55) < 0 ? (y + 100) : y;

            // set tooltip rect
            tooltipLine.style("opacity", 1);
            tooltipRectLine
                .attr("x", (tooltipX + 10) + "px")
                .attr("y", (tooltipY - 55) + "px");

            // set tooltip text
            if (hoveredData["nation"].length > 10) {
                tooltipRectLine.attr("width", 150)
            } else {
                tooltipRectLine.attr("width", 125)
            }

            utils.setTooltipText(tooltipTextLine, [
                `Nation: ${hoveredData["nation"]}`,
                `Confirmed: ${Math.round(hoveredData["confirmed"])}`,
                `New Case: ${hoveredData["newcase"]}`
            ], {x: tooltipX, y: tooltipY})
        } else {
            // if there is no hovered point
            // hide the tooltips
            tooltipLine.style("opacity", 0);
        }
    }

    // update tooltip for area chart
    function updateTooltipForArea(d) {
        const mousePos = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]};

        // adjust location
        mousePos.x = (mousePos.x + 65) > width ? (mousePos.x - 60) : mousePos.x;
        mousePos.y = (mousePos.y - 55) < 0 ? (mousePos.y + 100) : (mousePos.y + 50);

        // set tooltip rect
        tooltipArea.style("opacity", 1);
        tooltipRectArea
            .attr("x", (mousePos.x + 10) + "px")
            .attr("y", (mousePos.y - 55) + "px")
            .attr("width", d.key.length > 10 ? 150 : 100);

        // set tooltip text
        utils.setTooltipText(tooltipTextArea, [
            `Nation: ${d.key}`,
        ], mousePos)
    }

    // update tooltip for pie chart
    function updateTooltipForPie(mousePos, nation, percentage) {
        // set tooltip rect
        tooltipPie.style("opacity", 1);
        tooltipRectPie
            .attr("x", (mousePos.x + 10) + "px")
            .attr("y", (mousePos.y - 55) + "px");

        // set tooltip text
        utils.setTooltipText(tooltipTextPie, [
            `Nation: ${nation}`,
            `Percentage: ${percentage}`
        ], mousePos)
    }

    let hoveredDataPoint;

    // init all 3 plots
    function initComparisonPlots() {
        // read 3 data files
        d3.csv("data/covid19_timeseries.csv", overAllData => {
            d3.csv("data/covid19_nations_confirmed.csv", nationData => {
                d3.csv("data/covid19_nations_corr.csv", corData => {
                    // extract all dates
                    const dates = overAllData.map(row => utils.parseDate(row.date));
                    // set columns
                    const columns = ["date"].concat(mainNations).concat(["Others"]);

                    // extract data pairs for visualization
                    const dataObjs = utils.deepCopy(nationData).map(row => {
                        row["Others"] = utils.sum(Object.entries(row)
                            .filter(([key, _]) => !mainNations.includes(key) && key !== "date")
                            .map(([_, value]) => value));
                        row["date"] = utils.parseDate(row["date"]);
                        return Object.fromEntries(columns.map(key => [key, row[key]]));
                    });

                    // legend texts
                    const legendKeys = ["Australia", "Brazil", "China", "India", "Italy",
                        "Spain", "US", "United Kingdom", "Others"];
                    const legendText = ["Australia", "Brazil", "China", "India", "Italy",
                        "Spain", "US", "UK", "Others"];

                    // add legend in the bottom
                    utils.addLegendBottom(svgLegend, legendKeys, colorMap, legendText, false)
                    // add events for main nation legend
                    svgLegend.select("g.legend")
                        .selectAll("rect")
                        .each(function(d) {
                            // set colors
                            const rect = d3.select(this);
                            const color = colorMap[d];
                            const colorLight = d3.interpolate(color, "#FFF")(0.6);
                            const colorHiddenLight = d3.interpolate(color, colorLight)(0.5);
                            const colorDark = d3.interpolate(color, "#000")(0.8)

                            rect.on("mouseover", function() {
                                // if the mouse over the rect of legend
                                if (!legendIsFrozen) {
                                    if (rect.classed("hidden")) {
                                        rect.style("fill", colorHiddenLight);
                                    } else {
                                        rect.style("fill", colorLight);
                                    }
                                }
                            }).on("mouseout", function() {
                                // if the mouse move out of the rect of legend
                                if (!legendIsFrozen) {
                                    if (rect.classed("hidden")) {
                                        rect.style("fill", colorDark);
                                    } else {
                                        rect.style("fill", color);
                                    }
                                }
                            }).on("click", function() {
                                // if click the square
                                if (!legendIsFrozen) {
                                    // switch the rect class for filtering
                                    if (rect.classed("hidden")) {
                                        rect.classed("hidden", false);
                                        rect.style("fill", color);
                                        if (d === "Others") {
                                            selectOthers = true;
                                        } else {
                                            mainNationsSelection[d] = true;
                                        }
                                    } else {
                                        rect.classed("hidden", true)
                                        rect.style("fill", colorDark);
                                        if (d === "Others") {
                                            selectOthers = false;
                                        } else {
                                            mainNationsSelection[d] = false;
                                        }
                                    }
                                    // update all 3 plots with new selected nations
                                    updateComparisonPlots();
                                }
                            })
                        })

                    // init the area chart
                    function initComparisonLeft() {
                        const keys = ["Others"].concat(selectedNations);

                        // generate the stack data
                        const stacks = d3.stack()
                            .keys(keys)(dataObjs);

                        // add x axis
                        const x = d3.scaleUtc()
                            .domain(d3.extent(dates))
                            .range([0, width])
                        chartLeft.append("g")
                            .classed("x-axis", true)
                            .attr("transform", `translate(0, ${height})`)
                            .call(d3.axisBottom(x)
                                .tickFormat(d3.timeFormat('%b')));
                        // x label
                        chartLeft.append("text")
                            .text("Date")
                            .attr("transform", `translate(${width / 2}, ${height + 30})`)
                            .style("font-size", 14)
                            .style("text-anchor", "middle");

                        // add y axis
                        const y = d3.scaleLinear()
                            .domain([0, d3.max(stacks, d => d3.max(d, d => d[1]))])
                            .range([height, 0]);
                        chartLeft.append("g")
                            .classed("y-axis", true)
                            .call(d3.axisLeft(y).tickFormat(d => d / 1e6));
                        // y label
                        chartLeft.append("text")
                            .text("Confirmed (Million)")
                            .attr("transform", `translate(${-30}, ${height / 2}) rotate(270)`)
                            .style("font-size", 14)
                            .style("text-anchor", "middle");

                        // add area
                        const area = d3.area()
                            .x(d => x(d.data.date))
                            .y0(d => y(d[0]))
                            .y1(d => y(d[1]))

                        const areas = chartLeft.append("g")
                            .classed("stacked-area", true)
                            .selectAll()
                            .data(stacks).enter()
                            .append("path");

                        areas.attr("fill", ({key}) => key === "Others" ? othersColor : nationColors[key])
                            .transition()
                            .duration(500)
                            .attr("d", area);

                        // add events
                        areas.on("mouseover", function() {
                            // highlight the selection
                            areas.style("opacity", 0.5);
                            d3.select(this).style("opacity", 1);
                        })
                            // update the tooltip for area chart
                            .on("mousemove", updateTooltipForArea)
                            .on("mouseout", function() {
                                // hide the tooltip
                                areas.style("opacity", 1);
                                tooltipArea.style("opacity", 0);

                                tooltipRectArea
                                    .attr("x", -1000 + "px")
                                    .attr("y", -1000 + "px");

                                utils.setTooltipText(tooltipTextArea, [
                                    "",
                                ], {x: -1000, y: -1000})
                            })
                        // hide the tooltip if the mouse leave the svg
                        svgLeft.on("mouseleave", function() {
                            d3.selectAll("#comparison-left g.tooltip")
                                .style("opacity", 0);
                        });
                    }

                    // init the line chart
                    function initComparisonMid() {
                        const dataObjs = mainNations.map(nation => {
                            return {
                                nation: nation,
                                values: corData
                                    .filter(row => row["nation"] === nation
                                        && row["confirmed"] > 0
                                        && row["newcase"] > 0)
                                    .map(row => {
                                        return {
                                            nation: row["nation"],
                                            confirmed: +row["confirmed"],
                                            newcase: +row["newcase"]
                                        };
                                    })
                            };
                        });
                        // apply rollMean to smooth the lines
                        const nNeighbour = 15;

                        const smoothDataObjs = utils.deepCopy(dataObjs).map(({nation, values}) => {
                            const confirmedArr = values.map(each => each["confirmed"]);
                            const newcaseArr = values.map(each => each["newcase"]);

                            const smoothedConfirmedArr = utils.rollMean(confirmedArr, nNeighbour);
                            const smoothedNewcaseArr = utils.rollMean(newcaseArr, nNeighbour);

                            values = utils.range(values.length).map(index => {
                                return {
                                    nation: nation,
                                    confirmed: smoothedConfirmedArr[index],
                                    newcase: smoothedNewcaseArr[index]
                                }
                            })
                            return {nation, values};
                        });

                        const xs = corData.map(row => +row["confirmed"]);
                        const xMaxValue = Math.max(...xs);

                        const ys = corData.map(row => +row["newcase"]);
                        const yMaxValue = Math.max(...ys);

                        // add x axis
                        const x = d3.scaleLog()
                            .domain([10, xMaxValue])
                            .range([0, width]);

                        chartMid.append("g")
                            .classed("x-axis", true)
                            .attr("transform", `translate(0, ${height})`)
                            .call(d3.axisBottom(x).ticks(5).tickFormat(d => d));

                        chartMid.append("text")
                            .text("Confirmed")
                            .attr("transform", `translate(${width / 2}, ${height + 30})`)
                            .style("font-size", 14)
                            .style("text-anchor", "middle");

                        // add y axis
                        const y = d3.scaleLog()
                            .domain([1, yMaxValue])
                            .range([height, 0]);

                        chartMid.append("g")
                            .classed("y-axis", true)
                            .call(d3.axisLeft(y).ticks(4).tickFormat(d => d));

                        chartMid.append("text")
                            .text("New Case")
                            .attr("transform", `translate(-33, ${height / 2}) rotate(270)`)
                            .style("font-size", 14)
                            .style("text-anchor", "middle");

                        const line = d3.line()
                            .x(function(d) {
                                return x(d.confirmed)
                            })
                            .y(function(d) {
                                return y(d.newcase)
                            });

                        const paths = chartMid.append("g")
                            .classed("trends", true)
                            .classed("line-chart-element", true)
                            .style("stroke-width", 1)
                            .style("fill", "None")
                            .selectAll()
                            .data(smoothDataObjs)
                            .enter()
                            .append("path")
                            .attr("d", d => line(d.values))
                            .attr("stroke", d => colorMap[d.nation]);

                        const pointsData = smoothDataObjs.flatMap(nation => nation.values);
                        const originalPointsData = dataObjs.flatMap(nation => nation.values);
                        // get the point location for searching
                        const pointLoc = [];
                        pointsData.forEach((row, i) => {
                            pointLoc[i] = {
                                x: +x(row["confirmed"]),
                                y: +y(row["newcase"])
                            }
                        });

                        const points = chartMid.append("g")
                            .classed("points", true)
                            .classed("line-chart-element", true)
                            .selectAll()
                            .data(pointsData)
                            .enter()
                            .append("circle")
                            .attr("r", 5)
                            .attr("fill", d => colorMap[d.nation])
                            .attr("cx", (d, i) => pointLoc[i].x)
                            .attr("cy", (d, i) => pointLoc[i].y);

                        let closestPointIndex;
                        let hoveredPoint;

                        svgMid.on("mousemove", function() {
                            const mousePos = {x: d3.mouse(this)[0] - margin.left, y: d3.mouse(this)[1] - margin.top};
                            // calculate the distance
                            const distances = pointLoc
                                .map(eachPoint => Math.hypot(eachPoint.x - mousePos.x, eachPoint.y - mousePos.y))
                                // filter the distance threshold with 20 pixels
                                .map((distance, i) => {
                                    return {distance, i}
                                })
                                .filter(each => each.distance <= 30);

                            const closestPointDistance = utils.min(distances, each => each.distance);
                            closestPointIndex = utils.isNotNull(closestPointDistance) ? closestPointDistance.i : null;

                            // if there is a previous hovered point,
                            // reset the state
                            if (utils.isNotNull(hoveredDataPoint)) {
                                // un-class the hovered
                                d3.select(hoveredPoint)
                                    .classed("hovered", false);
                                hoveredDataPoint = null;
                            }

                            // add new hovered state for new hovered point
                            if (utils.isNotNull(closestPointIndex)) {
                                // class the new hovered, with color
                                hoveredPoint = points.nodes()[closestPointIndex];
                                d3.select(hoveredPoint).classed("hovered", true);
                                // record the data
                                hoveredDataPoint = originalPointsData[closestPointIndex]
                            }

                            updateTooltipForLine(hoveredDataPoint, hoveredPoint);

                        });

                        svgMid.on("mouseleave", function() {
                            d3.selectAll("#comparison-mid g.tooltip")
                                .style("opacity", 0);
                        });

                    }

                    // prepare data
                    nationDataLatest = Object.entries(nationData[overAllData.length - 1])
                        .filter(([key, value]) => key !== "date")
                        .map(([key, value]) => {
                            return {nation: key, value: +value};
                        });

                    sumValue = overAllData[overAllData.length - 1]["confirmed"];

                    // init the pie chart
                    function initComparisonRight() {
                        pieChartRight.selectAll()
                            .data([]).enter()
                            .append("path")
                            .classed("pie-chart-arc", true);

                        updatePieChart(nationDataLatest, sumValue, freezeLegend, arc, arcFocus, arcAnnotation);

                        const transitionIn = () => d3.transition()
                            .duration(250)
                            .ease(d3.easeQuadOut);

                        const transitionOut = () => d3.transition()
                            .duration(500)
                            .ease(d3.easeQuadOut);

                        const paths = pieChartRight
                            .selectAll("path.pie-chart-arc");

                        paths.on("mouseenter", function() {
                            const obj = this;
                            paths.filter(function() {
                                return this !== obj
                            }).style("opacity", 0.5);

                            d3.select(this)
                                .transition(transitionIn()).attr("d", arcFocus);

                        }).on("mouseout", function() {
                            paths.style("opacity", 1)
                                .transition(transitionOut()).attr("d", arc);

                            tooltipRectPie
                                .attr("x", -1000 + "px")
                                .attr("y", -1000 + "px");

                            utils.setTooltipText(tooltipTextPie, [
                                "", ""
                            ], {x: -1000, y: -1000})

                        }).on("mousemove", function(d) {
                            const mousePos = {x: d3.mouse(this)[0] + 100, y: d3.mouse(this)[1] + 150};

                            updateTooltipForPie(mousePos, d.data["nation"],
                                utils.round(d["value"] * 100 / sumValue, 2) + "%")
                        });
                    }

                    initComparisonLeft();
                    initComparisonRight();
                    setTimeout(() => {
                        initComparisonMid();
                    }, 500);
                });
            });
        });
    }

    let previousSelected = null;

    function getSelectedNations(mainNationFirst = true) {
        const selectedMainNations = Object.entries(mainNationsSelection)
            .filter(([_, value]) => value)
            .map(([key, _]) => key);

        return mainNationFirst ? selectedMainNations.concat(customNationsLegendKeys)
            : customNationsLegendKeys.concat(selectedMainNations);
    }

    function updateAreaChart() {
        selectedNations = getSelectedNations();
        // remove old axis
        const oldComponents = chartLeft.selectAll("g.x-axis, g.y-axis");

        d3.csv("data/covid19_nations_confirmed.csv", nationData => {
            const dates = nationData.map(row => utils.parseDate(row.date));
            const columns = ["date"].concat(selectedNations).concat(["Others"]);

            const dataObjs = utils.deepCopy(nationData).map(row => {
                row["Others"] = utils.sum(Object.entries(row)
                    .filter(([key, _]) => !selectedNations.includes(key) && key !== "date")
                    .map(([_, value]) => value));
                row["date"] = utils.parseDate(row["date"]);
                return Object.fromEntries(columns.map(key => [key, row[key]]));
            });

            const keys = selectOthers ? ["Others"].concat(selectedNations) : selectedNations;

            const stacks = d3.stack()
                .keys(keys)(dataObjs);

            // add x axis
            const x = d3.scaleUtc()
                .domain(d3.extent(dates))
                .range([0, width])
            chartLeft.append("g")
                .classed("x-axis", true)
                .attr("transform", `translate(0, ${height})`)
                .call(d3.axisBottom(x)
                    .tickFormat(d3.timeFormat('%b')));
            // add y axis
            const y = d3.scaleLinear()
                .domain([0, d3.max(stacks, d => d3.max(d, d => d[1]))])
                .range([height, 0]);
            chartLeft.append("g")
                .classed("y-axis", true)
                .call(d3.axisLeft(y).tickFormat(d => d / 1e6));
            // remove old axis
            oldComponents.remove();

            const area = d3.area()
                .x(d => x(d.data.date))
                .y0(d => y(d[0]))
                .y1(d => y(d[1]));

            const areas = chartLeft.select("g.stacked-area")
                .selectAll("path").data(stacks);

            areas.exit()
                .attr("fill", ({key}) => colorMap[key])
                .attr("d", area)
                .remove();

            areas.enter()
                .append("path")
                .attr("fill", ({key}) => colorMap[key])
                .transition()
                .duration(500)
                .attr("d", area);

            areas
                .attr("fill", ({key}) => colorMap[key])
                .transition()
                .duration(500)
                .attr("d", area);

            areas.on("mouseover", function() {
                areas.style("opacity", 0.5);
                d3.select(this).style("opacity", 1);
            }).on("mousemove", updateTooltipForArea)
                .on("mouseout", function() {
                    areas.style("opacity", 1);
                    tooltipArea.style("opacity", 0);

                    tooltipRectArea
                        .attr("x", -1000 + "px")
                        .attr("y", -1000 + "px");

                    utils.setTooltipText(tooltipTextArea, [
                        "",
                    ], {x: -1000, y: -1000})
                });
        });
    }

    function updateLineChart() {
        selectedNations = getSelectedNations();
        // remove old axis
        const oldAxis = chartMid.selectAll("g.x-axis, g.y-axis");
        const oldPaths = chartMid.selectAll("g.trends, g.points");

        d3.csv("data/covid19_nations_corr.csv", corData => {
            const dataObjs = selectedNations.map(nation => {
                return {
                    nation: nation,
                    values: corData
                        .filter(row => row["nation"] === nation
                            && row["confirmed"] > 0
                            && row["newcase"] > 0)
                        .map(row => {
                            return {
                                nation: row["nation"],
                                confirmed: +row["confirmed"],
                                newcase: +row["newcase"]
                            };
                        })
                };
            });
            // apply rollMean to smooth the lines
            const nNeighbour = 15;

            const smoothDataObjs = utils.deepCopy(dataObjs).map(({nation, values}) => {
                const confirmedArr = values.map(each => each["confirmed"]);
                const newcaseArr = values.map(each => each["newcase"]);

                const smoothedConfirmedArr = utils.rollMean(confirmedArr, nNeighbour);
                const smoothedNewcaseArr = utils.rollMean(newcaseArr, nNeighbour);

                values = utils.range(values.length).map(index => {
                    return {
                        nation: nation,
                        confirmed: smoothedConfirmedArr[index],
                        newcase: smoothedNewcaseArr[index]
                    }
                })
                return {nation, values};
            });

            const xs = corData.map(row => +row["confirmed"]);
            const xMaxValue = Math.max(...xs);

            const ys = corData.map(row => +row["newcase"]);
            const yMaxValue = Math.max(...ys);

            // add x axis
            const x = d3.scaleLog()
                .domain([10, xMaxValue])
                .range([0, width])

            chartMid.append("g")
                .classed("x-axis", true)
                .attr("transform", `translate(0, ${height})`)
                .call(d3.axisBottom(x).ticks(5).tickFormat(d => d));
            // add y axis
            const y = d3.scaleLog()
                .domain([1, yMaxValue])
                .range([height, 0]);
            chartMid.append("g")
                .classed("y-axis", true)
                .call(d3.axisLeft(y).ticks(4).tickFormat(d => d));

            oldAxis.remove();

            // add lines
            const line = d3.line()
                .x(function(d) {
                    return x(d.confirmed)
                })
                .y(function(d) {
                    return y(d.newcase)
                });

            const paths = chartMid.append("g")
                .classed("trends", true)
                .classed("line-chart-element", true)
                .style("stroke-width", 1)
                .style("fill", "None")
                .selectAll()
                .data(smoothDataObjs)
                .enter()
                .append("path")
                .attr("d", d => line(d.values))
                .attr("stroke", d => colorMap[d.nation]);

            oldPaths.remove();

            // add points
            const pointsData = smoothDataObjs.flatMap(nation => nation.values);
            const originalPointsData = dataObjs.flatMap(nation => nation.values);
            // get the point location for searching
            const pointLoc = [];
            pointsData.forEach((row, i) => {
                pointLoc[i] = {
                    x: +x(row["confirmed"]),
                    y: +y(row["newcase"])
                }
            });

            const points = chartMid.append("g")
                .classed("points", true)
                .classed("line-chart-element", true)
                .selectAll()
                .data(pointsData)
                .enter()
                .append("circle")
                .attr("r", 5)
                .attr("fill", d => colorMap[d.nation])
                .attr("cx", (d, i) => pointLoc[i].x)
                .attr("cy", (d, i) => pointLoc[i].y);

            let closestPointIndex;
            let hoveredPoint;

            // add events
            svgMid.on("mousemove", function() {
                const mousePos = {x: d3.mouse(this)[0] - margin.left, y: d3.mouse(this)[1] - margin.top};
                // calculate the distance
                const distances = pointLoc
                    .map(eachPoint => Math.hypot(eachPoint.x - mousePos.x, eachPoint.y - mousePos.y))
                    // filter the distance threshold with 20 pixels
                    .map((distance, i) => {
                        return {distance, i}
                    })
                    .filter(each => each.distance <= 30);

                const closestPointDistance = utils.min(distances, each => each.distance);
                closestPointIndex = utils.isNotNull(closestPointDistance) ? closestPointDistance.i : null;

                // if there is a previous hovered point,
                // reset the state
                if (utils.isNotNull(hoveredDataPoint)) {
                    // un-class the hovered
                    d3.select(hoveredPoint)
                        .classed("hovered", false);
                    hoveredDataPoint = null;
                }

                // add new hovered state for new hovered point
                if (utils.isNotNull(closestPointIndex)) {
                    // class the new hovered, with color
                    hoveredPoint = points.nodes()[closestPointIndex];
                    d3.select(hoveredPoint).classed("hovered", true);
                    // record the data
                    hoveredDataPoint = originalPointsData[closestPointIndex]
                }

                updateTooltipForLine(hoveredDataPoint, hoveredPoint);

            })
        })
    }

    function updatePieChartAnnotation(data, arcAnnotation, timeout) {
        pieChartRight.selectAll("g.annotation-group").transition().duration(300)
            .style("opacity", 0).remove();
        setTimeout(() => {
            const paths = pieChartRight.selectAll("path.pie-chart-arc");
            paths.each((d, i) => addPieChartAnnotation(arcAnnotation, data[i]));
        }, timeout);
    }

    function updatePieChart(nationDataLatest, sumValue, freezeLegend, arc, arcFocus, arcAnnotation) {
        const key = (d) => d.data["nation"];

        function findNeighborArc(i, data0, data1, key) {
            let d;
            return (d = findPreceding(i, data0, data1, key)) ? {
                    startAngle: d.endAngle,
                    endAngle: d.endAngle
                }
                : (d = findFollowing(i, data0, data1, key)) ? {
                        startAngle: d.startAngle,
                        endAngle: d.startAngle
                    }
                    : null;
        }

        // Find the element in data0 that joins the highest preceding element in data1.
        function findPreceding(i, data0, data1, key) {
            const m = data0.length;
            while (--i >= 0) {
                const k = key(data1[i]);
                for (let j = 0; j < m; ++j) {
                    if (key(data0[j]) === k) return data0[j];
                }
            }
        }

        // Find the element in data0 that joins the lowest following element in data1.
        function findFollowing(i, data0, data1, key) {
            const n = data1.length, m = data0.length;
            while (++i < n) {
                const k = key(data1[i]);
                for (let j = 0; j < m; ++j) {
                    if (key(data0[j]) === k) return data0[j];
                }
            }
        }

        function arcTween(d) {
            const i = d3.interpolate(this._current, d);
            this._current = i(0);
            return function(t) {
                return arc(i(t));
            };
        }

        // build selected data objects
        selectedNations = getSelectedNations();
        const selectedData = nationDataLatest.filter(d => selectedNations.includes(d["nation"]));
        if (selectOthers) {
            selectedData.push({
                nation: "Others",
                value: sumValue - utils.sum(selectedData.map(d => d.value))
            });
        }

        let paths = pieChartRight
            .selectAll("path.pie-chart-arc")

        const oldData = paths.data();
        const newData = pie(selectedData);
        const boundPaths = paths.data(newData, key);

        // perform updating
        boundPaths.enter()
            .append("path")
            .classed("pie-chart-arc", true)
            .attr("d", arc)
            .each(function(d, i) {
                this._current = findNeighborArc(i, oldData, newData, key) || d;
            })
            .attr("fill", function(d) {
                return colorMap[d.data["nation"]];
            });

        boundPaths.exit()
            .datum(function(d, i) {
                return findNeighborArc(i, oldData, newData, key) || d;
            })
            .transition()
            .duration(500)
            .attrTween("d", arcTween)
            .remove();

        boundPaths.transition()
            .duration(500)
            .attrTween("d", arcTween);

        // freeze the legend until the animation finishing
        freezeLegend(550);
        updatePieChartAnnotation(newData, arcAnnotation, 550);

        const transitionIn = () => d3.transition()
            .duration(250)
            .ease(d3.easeQuadOut);

        const transitionOut = () => d3.transition()
            .duration(500)
            .ease(d3.easeQuadOut);

        // add events after the animation finishing
        setTimeout(() => {
            paths = pieChartRight
                .selectAll("path.pie-chart-arc");

            paths.on("mouseenter", function() {
                const obj = this;
                paths.filter(function() {
                    return this !== obj
                }).style("opacity", 0.5);

                d3.select(this)
                    .transition(transitionIn()).attr("d", arcFocus);

            }).on("mouseout", function() {
                paths.style("opacity", 1)
                    .transition(transitionOut()).attr("d", arc);

                tooltipRectPie
                    .attr("x", -1000 + "px")
                    .attr("y", -1000 + "px");

                utils.setTooltipText(tooltipTextPie, [
                    "", ""
                ], {x: -1000, y: -1000})

            }).on("mousemove", function(d) {
                const mousePos = {x: d3.mouse(this)[0] + 100, y: d3.mouse(this)[1] + 150};

                updateTooltipForPie(mousePos, d.data["nation"],
                    utils.round(d["value"] * 100 / sumValue, 2) + "%")
            });
        }, 550);

    }

    // update for new selected nations
    function updateComparisonPlots() {
        updatePieChart(nationDataLatest, sumValue, freezeLegend, arc, arcFocus, arcAnnotation);
        updateAreaChart();
        updateLineChart();
    }

    // update for re-select selection bars
    function updatePlots() {
        updateMap();
        updateLineChart();
    }

    initComparisonPlots();
    initMap();
    document.getElementById("nation-world-map-fill1").onchange = updatePlots;
    document.getElementById("nation-world-map-scale1").onchange = updatePlots;
}

runNationSection()