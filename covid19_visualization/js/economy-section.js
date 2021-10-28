// run the economy section
function runEconomySection() {
    // init
    const svgWidth = +d3.select("#economy-left").attr("width").slice(0, -2);
    const svgHeight = +d3.select("#economy-left").attr("height").slice(0, -2);

    const margin = {top: 20, right: 100, bottom: 30, left: 45};
    const width = svgWidth - margin.left - margin.right;
    const height = svgHeight - margin.top - margin.bottom;

    // svg selections
    const svgLeft = d3.select("#economy-left");
    const svgRight = d3.select("#economy-right");
    const svgMap = d3.select("#economy-world-map");
    const svgColorBar = d3.select("#economy-color-bar");
    const descriptionContainer = d3.select("#economy-description-container")

    // init color bar <g> tag
    const colorBar = svgColorBar.append("g")
        .attr("id", "economy-color-bar-g")
        .classed("color-bar", true)
        .attr("transform", "translate(0, 10)");

    // map components
    const projection = d3.geoMercator().scale(90).translate([280, 145]);
    const path = d3.geoPath().projection(projection);

    const scatterLeft = svgLeft
        .append("g")
        .classed("scatter", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const scatterRight = svgRight
        .append("g")
        .classed("scatter", true)
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // color maps related to the elements
    const colorMap = {point: "#F78C6C", baseline: "#FFCB6B", regression: "#89DDFF"};

    let mapColor2020, mapColor2021;

    // function for adding regression line
    function addRegressionLine(scatter, xScale, yScale, year, fill, xMinValue, xMaxValue) {
        const file = `data/sample_${fill}_${year}.csv`;

        d3.csv(file, data => {
            data = data.filter(row => row["sample"] <= xMaxValue && row["sample"] >= xMinValue);
            // Show confidence interval
            scatter.append("path")
                .classed("regression-interval", true)
                .datum(data)
                .attr("fill", "#616161")
                .attr("stroke", "none")
                .style("opacity", 0.2)
                .attr("d", d3.area()
                    .x(function(d) {
                        return xScale(d["sample"])
                    })
                    .y0(function(d) {
                        return yScale(d["lwr"])
                    })
                    .y1(function(d) {
                        return yScale(d["upr"])
                    })
                )
            // Add the line
            scatter
                .append("path")
                .classed("regression-line", true)
                .datum(data)
                .attr("fill", "none")
                .attr("stroke", colorMap["regression"])
                .attr("stroke-width", 1.5)
                .attr("d", d3.line()
                    .x(function(d) {
                        return xScale(d["sample"])
                    })
                    .y(function(d) {
                        return yScale(d["fit"])
                    })
                )
        })
    }

    // update regression line
    function updateRegressionLine(scatter, xScale, yScale, year, fill, xMinValue, xMaxValue) {
        const file = `data/sample_${fill}_${year}.csv`;

        d3.csv(file, data => {
            data = data.filter(row => row["sample"] <= xMaxValue && row["sample"] >= xMinValue);
            // Show confidence interval
            scatter.select("path.regression-interval")
                .datum(data)
                .transition()
                .attr("fill", "#616161")
                .attr("stroke", "none")
                .style("opacity", 0.2)
                .attr("d", d3.area()
                    .x(function(d) {
                        return xScale(d["sample"])
                    })
                    .y0(function(d) {
                        return yScale(d["lwr"])
                    })
                    .y1(function(d) {
                        return yScale(d["upr"])
                    })
                )
            // Add the line
            scatter.select("path.regression-line")
                .datum(data)
                .transition()
                .attr("fill", "none")
                .attr("stroke", colorMap["regression"])
                .attr("stroke-width", 1.5)
                .attr("d", d3.line()
                    .x(function(d) {
                        return xScale(d["sample"])
                    })
                    .y(function(d) {
                        return yScale(d["fit"])
                    })
                )
        })
    }

    // set axis with given elements
    function setAxis(data, chart, xs, ys, xMinValue, xMaxValue) {
        // Add X axis
        const x = d3.scaleLog()
            .domain([xMinValue, xMaxValue])
            .range([0, width]);

        chart.append("g")
            .classed("axis", true)
            .classed("x-axis", true)
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(x).ticks(2).tickFormat(d => d));

        // Add Y axis
        const y = d3.scaleLinear()
            .domain([Math.min(...ys.filter(utils.isNotNull)) - 0.5,
                Math.max(0.5, Math.max(...ys.filter(utils.isNotNull)))])
            .range([height, 0]);

        chart.append("g")
            .classed("axis", true)
            .classed("y-axis", true)
            .call(d3.axisLeft(y));

        return {x, y}
    }

    // generate proper text for legend
    function fill2legendText(fill) {
        switch (fill) {
            case "gdp_rate":
                return "GDP Rate";
            case "investment":
                return "Investment";
            case "demand":
                return "Demand";
        }
    }

    // filter the validated data (not null)
    function filterValidData(geoJson, oecdKey, returnIndex = false) {
        return geoJson.features.map((each, i) => {
            return {value: each.properties, i: i};
        }).filter(each => {
            const xs = geoJson.features.map(each => +each.properties["confirmed_per_capita"]);
            const oecdObj = geoJson.features.map(each => each.properties[oecdKey]);
            return utils.isNotZero(xs[each.i]) && utils.isNotNull(oecdObj[each.i])
        }).map(each => returnIndex ? each : each.value);
    }

    // wrangling data and build axis
    function preprocessingAndBuildAxis(geoJson, year, scatter) {
        const fill = document.getElementById("economy-world-map-fill").value;
        const oecdKey = year === 2020 ? "oecd_2020" : "oecd_2021";

        const data = filterValidData(geoJson, oecdKey);

        const xs = data.map(each => +each["confirmed_per_capita"]);
        const ys = data.map(each => each[oecdKey][fill]);
        const xMinValue = Math.min(...xs.filter(utils.isNotZero)) * 0.9;
        const xMaxValue = Math.max(...xs.filter(utils.isNotZero)) * 1.1;

        const {x, y} = setAxis(data, scatter, xs, ys, xMinValue, xMaxValue);

        // get the point location for searching
        const pointLoc = [];
        data.forEach((row, i) => {
            pointLoc[i] = {
                x: +x(row["confirmed_per_capita"]),
                y: +y(row[oecdKey][fill])
            }
        })

        return {data, pointLoc, xMinValue, xMaxValue, x, y, fill, oecdKey};
    }

    let selected, hovered;

    const removeElementsClassAndGetNew = (clazz, color) => () => {
        const fill = document.getElementById("economy-world-map-fill").value;

        // world map
        const mapPaths = svgMap.selectAll("path");
        mapPaths.classed(clazz, false);
        // left scatter
        const leftCircles = svgLeft.select("g.scatter g.points").selectAll("circle");
        leftCircles.classed(clazz, false)
            .style("fill", function(d) {
                if (d3.select(this).classed(clazz)) {
                    return color;
                } else {
                    return mapColor2020(null, d["oecd_2020"][fill], false);
                }
            });
        // right scatter
        const rightCircles = svgRight.select("g.scatter g.points").selectAll("circle");
        rightCircles.classed(clazz, false)
            .style("fill", function(d) {
                if (d3.select(this).classed(clazz)) {
                    return color;
                } else {
                    return mapColor2021(null, d["oecd_2021"][fill], false);
                }
            });

        return [mapPaths, leftCircles, rightCircles];
    }

    const removeSelectedElementsAndSelectNew = removeElementsClassAndGetNew("selected", "#C792EA");

    // function of adding legend for scatter
    function addLegendForScatter(chart, fill) {
        const keys = ["point", "regression", "baseline"]
        return utils.addLegendRight(chart, width, keys, colorMap,
            [fill2legendText(fill), "Regression", "Baseline"], false,
            function() {
                chart.append("g")
                    .classed("legend", true)
                    .selectAll()
                    .data(keys)
                    .enter()
                    .each(function(d, i) {
                        // add legend symbols
                        switch (d) {
                            case "point":
                                d3.select(this)
                                    .append("circle")
                                    .classed("legend-circle", true)
                                    .attr("cx", width + 30)
                                    .attr("cy", 110 + i * 25)
                                    .attr("r", 10)
                                    .style("fill", colorMap["point"]);
                                break;
                            default:
                                d3.select(this)
                                    .append("rect")
                                    .classed("legend-rect", true)
                                    .attr("x", width + 20)
                                    .attr("y", 100 + i * 25)
                                    .attr("width", 20)
                                    .attr("height", 20)
                                    .style("fill", d => colorMap[d]);
                        }
                    })
            })
    }

    // events for scatter interactions
    function addScatterInteractiveEvents(svg, pointLoc, mapColor, oecdKey, fill, points, data) {
        let hoveredPoint = null;
        let closestPointIndex = null;

        svg.on("mousemove", function() {
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
            if (utils.isNotNull(hovered)) {
                // un-class the hovered
                d3.select(hoveredPoint)
                    .classed("hovered", false)
                    .style("fill", function(d) {
                        if (d3.select(this).classed("selected")) {
                            return "#C792EA";
                        } else {
                            return mapColor(null, d[oecdKey][fill], false);
                        }
                    });

                // reset the color in world map for previous hovered nation
                svgMap.selectAll("path")
                    .each(function(d, i) {
                        if (d.properties["nation"] === hovered["nation"]) {
                            const oecdKey = document.getElementById("economy-world-map-year").value;
                            const mapColor = oecdKey === "oecd_2020" ? mapColor2020 : mapColor2021;
                            d3.select(this).style("fill", mapColor(i, getOecdValue(d, oecdKey, fill), false));
                        }
                    })
                hovered = null;
            }

            // add new hovered state for new hovered point
            if (utils.isNotNull(closestPointIndex)) {
                // class the new hovered, with color
                hoveredPoint = points.nodes()[closestPointIndex];
                d3.select(hoveredPoint)
                    .classed("hovered", true)
                    .style("fill", function() {
                        if (d3.select(this).classed("selected")) {
                            return "#C792EA";
                        } else {
                            return "#89DDFF";
                        }
                    });
                // color the hovered nation in world map
                hovered = data[closestPointIndex]
                svgMap.selectAll("path")
                    .each(function(d) {
                        if (d.properties["nation"] === hovered["nation"]) {
                            d3.select(this).style("fill", "#89DDFF");
                        }
                    });
            }
        }).on("click", function() {
            if (utils.isNotNull(hovered)) {
                // remove the previous select
                // select the elements by the index
                const [mapPath, leftCircle, rightCircle] = removeSelectedElementsAndSelectNew().map(elements => {
                    let res = null;
                    elements.each(function(d) {
                        if (utils.isNotNull(d.properties)) {
                            if (d.properties["nation"] === hovered["nation"]) {
                                res = this;
                            }
                        } else {
                            if (d["nation"] === hovered["nation"]) {
                                res = this;
                            }
                        }
                    })
                    return res;
                });

                // add visual effect to new selected one
                [mapPath, leftCircle, rightCircle].forEach(element => {
                    d3.select(element).classed("selected", true)
                        .style("fill", "#C792EA");
                });

                // assign the new data to `selected` variable
                selected = hovered;
                updateDescription();

                // update the annotation
                updateAnnotations(leftCircle, rightCircle)

                // display nation name
                d3.select("#economy-world-map-bottom-text")
                    .html(`Selected Country: ${selected["nation"]}`);
            }
        });
    }

    // the function for initializing scatter
    function initScatter(svg, scatter, year) {

        d3.json("data/geojson.json", geoJson => {
            const {data, pointLoc, xMinValue, xMaxValue, x, y, fill, oecdKey} =
                preprocessingAndBuildAxis(geoJson, year, scatter);

            // add subtitle
            scatter.append("text")
                .text(year)
                .attr("transform", `translate(${width / 2}, -5)`)
                .style("font-size", 14)
                .style("text-anchor", "middle");

            // add x-axis label
            scatter.append("text")
                .text("Confirmed Per Capita")
                .attr("transform", `translate(${width / 2}, ${height + 30})`)
                .style("font-size", 14)
                .style("text-anchor", "middle");

            // add y-axis label
            scatter.append("text")
                .text("Growth Rate")
                .attr("transform", `translate(${-30}, ${height / 2}) rotate(270)`)
                .style("font-size", 14)
                .style("text-anchor", "middle");

            // add regression line
            addRegressionLine(scatter, x, y, year, fill, xMinValue, xMaxValue);
            // add horizontal baseline
            utils.addHorizontalBaseLine(scatter, 0, x, y, xMaxValue, xMinValue, "#FFCB6B");
            // add legend
            addLegendForScatter(scatter, fill);
            // add points
            const mapColor = year === 2020 ? mapColor2020 : mapColor2021;
            const points = scatter.append("g")
                .classed("points", true)
                .selectAll()
                .data(data)
                .enter()
                .append("circle")
                .attr("cx", (d, i) => +pointLoc[i]["x"])
                .attr("cy", (d, i) => +pointLoc[i]["y"])
                .attr("r", 3)
                .style("fill", (d, i) => mapColor(i, d[oecdKey][fill], true));
            // add events
            addScatterInteractiveEvents(svg, pointLoc, mapColor, oecdKey, fill, points, data);
        });
    }

    // the function for updating scatter
    function updateScatter(svg, scatter, year) {
        // select old axis for removing
        const oldAxis = scatter.selectAll(".axis");

        // read geoJson
        d3.json("data/geojson.json", geoJson => {
            const {data, pointLoc, xMinValue, xMaxValue, x, y, fill, oecdKey} =
                preprocessingAndBuildAxis(geoJson, year, scatter);
            oldAxis.remove();

            const mapColor = year === 2020 ? mapColor2020 : mapColor2021;
            // Update points
            const points = scatter.selectAll('g.points circle')
                .transition()
                .attr("cx", (d, i) => pointLoc[i]["x"])
                .attr("cy", (d, i) => pointLoc[i]["y"])
                .style("fill", function(d, i) {
                    if (d3.select(this).classed("selected")) {
                        return "#C792EA";
                    } else {
                        return mapColor(i, getOecdValue(d, oecdKey, fill), true);
                    }
                })
                .attr("r", d => utils.isNotNull(getOecdValue(d, oecdKey, fill)) ? 3 : 0);

            // update regression line
            updateRegressionLine(scatter, x, y, year, fill, xMinValue, xMaxValue);
            // Update horizontal baseline
            utils.updateHorizontalBaseLine(scatter, 0, x, y, xMaxValue, xMinValue);
            utils.updateLegendText(scatter, width, ["point", "regression", "baseline"], colorMap,
                [fill2legendText(fill), "Regression", "Baseline"]);
            // add events
            addScatterInteractiveEvents(svg, pointLoc, mapColor, oecdKey, fill, points, data);
        });
    }

    function updateColorBar(colorBar, colorScale, valueMin, valueMax) {
        const scale = document.getElementById("economy-world-map-scale").value;
        const fill = document.getElementById("economy-world-map-fill").value;
        utils.addColorBar(colorBar, colorScale, valueMin, valueMax, scale, fill, 250);
    }

    // curried function to build color map
    function fill2color(fill, scale, oecdKey, geoJson) {
        // extract values for fill color
        const result = filterValidData(geoJson, oecdKey, true);
        const data = result.map(each => each.value);
        const index = result.map(each => each.i);
        const values = data.map(x => +x[oecdKey][fill]);

        // min and max value of the array
        const maxValue = Math.max(...values.filter(utils.isNotZero).filter(utils.isNotNull))
        const minValue = Math.min(...values.filter(utils.isNotZero).filter(utils.isNotNull))
        // calculate the sorted orders
        const orders = utils.argsort(data.map(x => +x[oecdKey][fill]));

        let value2range;
        let colorBarMax, colorBarMin;
        // get value2range function for color map
        switch (scale) {
            case "linear":
                colorBarMax = maxValue;
                colorBarMin = minValue;
                value2range = d3.scaleLinear()
                    .domain([colorBarMin, colorBarMax])
                    .range([0, 1]);
                break;
            case "log":
                colorBarMax = maxValue;
                colorBarMin = minValue;
                value2range = d3.scaleLog()
                    .domain([colorBarMin, colorBarMax])
                    .range([0, 1]);
                break;
            case "percentile":
                colorBarMax = 100;
                colorBarMin = 0;
                value2range = d3.scaleLinear()
                    .domain([colorBarMin, colorBarMax])
                    .range([0, 1]);
                break;
        }

        if (oecdKey === document.getElementById("economy-world-map-year").value) {
            updateColorBar(colorBar, x => d3.interpolateRdYlGn(value2range(x)),
                colorBarMin, colorBarMax);
        }

        // if the value is 0, use "gray" color, else use the color generated
        return (i, value, fromScatter = false) => {
            if (value === null) {
                return "gray";
            } else if (scale === "percentile") {
                if (i === null) {
                    i = values.indexOf(value);
                    return d3.interpolateRdYlGn(value2range(orders[i] / orders.length * 100))
                } else {
                    return fromScatter ? d3.interpolateRdYlGn(value2range(orders[i] / orders.length * 100)) :
                        d3.interpolateRdYlGn(value2range(orders[index.indexOf(i)] / orders.length * 100));
                }
            } else {
                return d3.interpolateRdYlGn(value2range(value));
            }
        };
    }

    // a helper function for get nullable oecd values
    function getOecdValue(row, oecdKey, fill) {
        const oecdObj = utils.isNotNull(row.properties) ? row.properties[oecdKey] : row[oecdKey];
        if (utils.isNotNull(oecdObj)) {
            return oecdObj[fill];
        } else {
            return null;
        }
    }

    // a function for adding map interaction events
    function addMapInteractiveEvents(paths, mapColor, oecdKey, fill) {
        let selectedPath = null;

        // assign the interactivity for selected and hovered
        paths.on("mouseover", function(row) {
            const element = d3.select(this);
            // color the world map nation
            element.classed("hovered", true)
                .style("fill", "#89DDFF");

            // change scatter point's color
            [svgLeft, svgRight].forEach(svg => {
                svg.select("g.scatter g.points").selectAll("circle")
                    .filter(d => d["nation"] === row.properties["nation"])
                    .classed("hovered", true)
                    .style("fill", function() {
                        if (d3.select(this).classed("selected")) {
                            return "#C792EA";
                        } else {
                            return "#89DDFF";
                        }
                    });
            })
        })
            .on("mouseout", function(row, i) {
                // when the mouse move out, reverse the status
                const element = d3.select(this);
                const newColor = mapColor(i, getOecdValue(row, oecdKey, fill), false);
                element.classed("hovered", false)
                    .style("fill", newColor);

                const newColorLeft = mapColor2020(i, getOecdValue(row, "oecd_2020", fill), false);
                const newColorRight = mapColor2021(i, getOecdValue(row, "oecd_2021", fill), false);

                svgLeft.select("g.scatter g.points").selectAll("circle")
                    .classed("hovered", false)
                    .filter(d => d["nation"] === row.properties["nation"])
                    .style("fill", function() {
                        if (d3.select(this).classed("selected")) {
                            return "#C792EA";
                        } else {
                            return newColorLeft;
                        }
                    });

                svgRight.select("g.scatter g.points").selectAll("circle")
                    .classed("hovered", false)
                    .filter(d => d["nation"] === row.properties["nation"])
                    .style("fill", function() {
                        if (d3.select(this).classed("selected")) {
                            return "#C792EA";
                        } else {
                            return newColorRight;
                        }
                    });
            })
            .on("click", function(row) {
                // display nation name
                d3.select("#economy-world-map-bottom-text")
                    .html(`Selected Country: ${row.properties["nation"]}`);
                // remove old selected
                const [, leftCircles, rightCircles] = removeSelectedElementsAndSelectNew();
                // mark the selected nation in world map
                selected = row.properties;
                updateDescription();

                d3.select(this).classed("selected", true);
                selectedPath = this;

                // mark the selected nation in scatters
                const selectedCircles = [];
                [leftCircles, rightCircles].forEach(circles => {
                    circles.each(function(d) {
                        if (d["nation"] === selected["nation"]) {
                            selectedCircles.push(this);
                            const element = d3.select(this);
                            element.classed("selected", true)
                                .style("fill", "#C792EA");
                        }
                    })
                })
                // update annotations
                updateAnnotations(...selectedCircles);
            });
    }

    async function initMap() {
        // collect variables
        const fill = document.getElementById("economy-world-map-fill").value;
        const oecdKey = document.getElementById("economy-world-map-year").value;
        const scale = document.getElementById("economy-world-map-scale").value;

        await new Promise((resolve) => {
            // read geoJson
            d3.json("data/geojson.json", geoJson => {
                // build the color map for selected fill value
                mapColor2020 = fill2color(fill, scale, "oecd_2020", geoJson);
                mapColor2021 = fill2color(fill, scale, "oecd_2021", geoJson);
                resolve(true);

                const mapColor = oecdKey === "oecd_2020" ? mapColor2020 : mapColor2021;

                // build paths with world-map-nation class
                const paths = svgMap.selectAll("path")
                    .data(geoJson.features).enter()
                    .append("path")
                    .classed("world-map-nation", true)
                    .attr("d", path)
                    .style("fill", (row, i) => mapColor(i, getOecdValue(row, oecdKey, fill), false));

                // add interactive events
                addMapInteractiveEvents(paths, mapColor, oecdKey, fill);
            })
        });
    }

    const updateMap = (onlyUpdateYear) => async () => {
        // collect variables
        const fill = document.getElementById("economy-world-map-fill").value;
        const scale = document.getElementById("economy-world-map-scale").value;
        const oecdKey = document.getElementById("economy-world-map-year").value;

        await new Promise((resolve) => {
            // read geoJson
            d3.json("data/geojson.json", geoJson => {
                // build the color map for selected fill value
                if (!onlyUpdateYear) {
                    mapColor2020 = fill2color(fill, scale, "oecd_2020", geoJson);
                    mapColor2021 = fill2color(fill, scale, "oecd_2021", geoJson);
                } else {
                    fill2color(fill, scale, oecdKey, geoJson);
                }
                resolve(true);
                const mapColor = oecdKey === "oecd_2020" ? mapColor2020 : mapColor2021;
                const paths = svgMap.selectAll("path");

                // assign the interactivity for selected
                paths.transition()
                    .style("fill", (row, i) => mapColor(i, getOecdValue(row, oecdKey, fill), false));

                addMapInteractiveEvents(paths, mapColor, oecdKey, fill);
            });
        });
    }

    // build annotations for selected point
    function buildAnnotations(circle) {
        const annotation = [];
        d3.select(circle).each(function(d) {
            const element = d3.select(this);

            if (+element.attr("r") !== 0) {
                const x = element.attr("cx");
                const y = element.attr("cy");
                const dx = x + 40 > width - 100 ? -40 : 40;
                const dy = y - 50 < 0 ? 50 : -50;

                annotation.push(
                    new utils.Annotation("", d["nation"], x, y, dx, dy, "#C792EA", 200)
                );
            }
        })
        return annotation;
    }

    // function for removing annotations for scatters
    function removeAnnotationsForScatter() {
        utils.removeAnnotations(
            "#economy-left g.scatter g.annotation-group",
            "#economy-right g.scatter g.annotation-group"
        )
    }

    function updateAnnotations(leftCircle, rightCircle) {
        // remove all previous annotations
        removeAnnotationsForScatter();

        const annotationLeft = buildAnnotations(leftCircle);
        const annotationRight = buildAnnotations(rightCircle);

        scatterLeft.append("g")
            .classed("annotation-group", true)
            .call(utils.makeAnnotations(annotationLeft));

        scatterRight.append("g")
            .classed("annotation-group", true)
            .call(utils.makeAnnotations(annotationRight));
    }

    // update description block in the section
    function updateDescription() {
        // read variables
        const fill = document.getElementById("economy-world-map-fill").value;
        const oecdKey = document.getElementById("economy-world-map-year").value;

        const mapColor = oecdKey === "oecd_2020" ? mapColor2020 : mapColor2021;

        // if there is a nation selected
        if (utils.isNotNull(selected)) {
            const oecdValue = getOecdValue(selected, oecdKey, fill);
            const color = mapColor(null, oecdValue, false);
            // change the block border of the description
            descriptionContainer
                .transition()
                .style("border-color", color);

            // update the html
            descriptionContainer.html(`<p style="color:${color}; font-size:20pt">${selected.nation}</p>
<p style="float:right; text-align: right">${selected["confirmed"]}</p>
<p>Confirmed:</p>
<p>Confirmed Per Millions:</p>
<p style="text-align: right">${Math.round(selected["confirmed_per_capita"] * 1e6)}</p>
<p style="float:right; text-align: right">${utils.isNotNull(oecdValue) ? utils.round(oecdValue, 4) : "Not Available"}</p>
<p>Growth Rate:</p>
`)
        }

    }

    // synchronize the loading process
    // build map
    initMap().then(() => {
        // build scatters
        initScatter(svgLeft, scatterLeft, 2020);
        initScatter(svgRight, scatterRight, 2021);
    });

    // assign the update events
    document.getElementById("economy-world-map-fill").onchange = () => {
        removeAnnotationsForScatter();
        updateMap(false)().then(() => {
            // wait for colorMap loaded
            updateScatter(svgLeft, scatterLeft, 2020);
            updateScatter(svgRight, scatterRight, 2021);
            updateDescription();
        });
    };

    document.getElementById("economy-world-map-scale").onchange = () => {
        updateMap(false)().then(() => {
            // wait for colorMap loaded
            updateScatter(svgLeft, scatterLeft, 2020);
            updateScatter(svgRight, scatterRight, 2021);
            updateDescription();
        });
    };
    document.getElementById("economy-world-map-year").onchange = updateMap(true);
}

runEconomySection()
