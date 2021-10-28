// this utils is about the util functions used in the project
const utils = {
    parseDate: d3.timeParse("%Y-%m-%d"),

    // JS implementation of numpy.argsort
    argsort: function(values) {
        // the ordered index of the array
        return values.map((value, index) => {
            return {value, index}
        })
            .sort((a, b) => a.value - b.value)
            .map(({value, index}, order) => {
                return {index, order}
            })
            // reorder the indexes
            .sort((a, b) => a.index - b.index)
            .map(each => each.order);
    },

    min: function(arr, callback = x => x) {
        const values = arr.map(callback)
        const minValue = Math.min(...values);
        return arr[values.indexOf(minValue)];
    },

    sum: function(arr, callback = x => +x) {
        return this.isNotZero(arr.length) ? arr.map(callback).reduce((a, b) => a + b)
            : 0;
    },

    mean: function(arr, callback = x => +x) {
        return this.sum(arr, callback) / arr.length;
    },

    // JS implementation of range
    range: function(n) {
        return [...Array(n).keys()];
    },

    round: function(num, n) {
        const scale = 10 ** n;
        return Math.round(num * scale) / scale;
    },

    // roll mean implementation by JS
    rollMean: function(arr, n = 5) {
        return arr.map((each, i) => this.getNeighbours(arr, i, n))
            .map(each => this.mean(each))
    },

    getNeighbours: function(arr, i, n = 5) {
        const margin = Math.floor(n / 2);
        const lowerIndex = Math.max(0, i - margin);
        const upperIndex = Math.min(arr.length, i + margin + 1);
        return arr.slice(lowerIndex, upperIndex);
    },

    isNotNull: function(x) {
        return x !== null && x !== undefined;
    },

    isNotZero: function(x) {
        return +x !== 0;
    },

    selectBy: function(arr, selected, callback) {
        const values = arr.map(callback);

        return selected
            .map(each => values.indexOf(each))
            .map(i => arr[i])
    },

    // Melt the data as an array of {x, y} tuples
    melt: function(categories, data) {
        return categories.map(category => {
            return {
                category: category,
                values: data.map(function(row) {
                    return {
                        category: category,
                        date: utils.parseDate(row["date"]),
                        value: +row[category]
                    };
                })
            };
        });
    },

    // an implementation for deep copy
    deepCopy: function(obj) {
        if (Array.isArray(obj)) {
            const newArr = [];
            obj.forEach(value => {
                newArr.push(this.deepCopy(value));
            })
            return newArr;
        } else if (typeof obj === "object") {
            const newObj = {};
            Object.entries(obj).forEach(([key, value]) => {
                newObj[key] = this.deepCopy(value);
            })
            return newObj;
        } else {
            return obj;
        }
    },

    // a helper function for building tooltip
    initTooltip: function(svg, nRow, width) {
        // prepare tooltip
        const tooltip = svg.append("g")
            .attr("class", "tooltip")
            .style("opacity", 0);

        const tooltipRect = tooltip.append("rect")
            .attr("class", "tooltip-rect")
            .attr("height", 5 + nRow * 15)
            .attr("width", width)
            .attr("fill", "black")
            .style("stroke", "white")
            .attr("rx", "5px")
            .attr("ry", "5px");

        const tooltipText = tooltip.append("text")
            .attr("class", "tooltip-text")
            .style("font-size", 12);

        this.range(nRow).map((n) => tooltipText.append("tspan").attr("class", `tooltip-text-row${n + 1}`))
        return {tooltip, tooltipRect, tooltipText}
    },

    setTooltipText: function(tooltipText, textArr, mousePos) {
        return textArr.forEach(function(line, i) {
            tooltipText
                .select(`.tooltip-text-row${i + 1}`)
                .text(line)
                .attr("x", (mousePos.x + 15.) + "px")
                .attr("y", (-40 + mousePos.y + 15 * i) + "px")
                .attr("text-anchor", "left");
        })
    },

    // find closest value in ordered array
    // using binary search
    findClosest: function(array, target) {
        const n = array.length;

        if (target <= array[0])
            return array[0];
        if (target >= array[n - 1])
            return array[n - 1];

        function getClosest(val1, val2) {
            if (target - val1 >= val2 - target)
                return val2;
            else
                return val1;
        }

        // Doing binary search
        let low = 0, high = n, mid = 0;
        while (low < high) {
            mid = Math.floor((low + high) / 2)
            if (array[mid] === target)
                return array[mid];
            if (target < array[mid]) {
                if (mid > 0 && target > array[mid - 1])
                    return getClosest(array[mid - 1], array[mid], target);
                high = mid;
            } else {
                if (mid < n - 1 && target < array[mid + 1])
                    return getClosest(array[mid], array[mid + 1], target);
                low = mid + 1;
            }
        }
        return array[mid];
    },

    buildXAxisWithDate: function(chart, data, width, height) {
        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => this.parseDate(d.date)))
            .range([0, width]);
        // generate sticks
        chart.append("g")
            .attr("transform", `translate(0, ${height})`)
            .call(d3.axisBottom(x)
                .tickFormat(d3.timeFormat('%b')));
        // add x label
        chart.append("text")
            .text("Date")
            .attr("transform", `translate(${width / 2}, ${height + 30})`)
            .style("font-size", 12)
            .style("text-anchor", "middle")
        return x;
    },

    initTooltipLine: function(pos, chart, y, yMaxValue, yMinValue = 0) {
        return chart.append("line")
            .classed("tooltip", true)
            .classed(`tooltip-line-${pos}`, true)
            .attr("y1", y(yMaxValue))
            .attr("y2", y(yMinValue))
            .attr("x1", 150)
            .attr("x2", 150)
            .style("stroke", "white")
            .style("opacity", 0);
    },

    addHorizontalBaseLine: function(chart, y, xScale, yScale, xMaxValue, xMinValue, color) {
        return chart.append("line")
            .classed("baseline", true)
            .attr("y1", yScale(y))
            .attr("y2", yScale(y))
            .attr("x1", xScale(xMaxValue))
            .attr("x2", xScale(xMinValue))
            .style("stroke-dasharray", "5, 5")
            .style("stroke", color)
            .style("stroke-width", 2)
            .style("opacity", 1)
    },
    updateHorizontalBaseLine: function(chart, y, xScale, yScale, xMaxValue, xMinValue) {
        return chart.select(".baseline")
            .transition()
            .attr("y1", yScale(y))
            .attr("y2", yScale(y))
            .attr("x1", xScale(xMaxValue))
            .attr("x2", xScale(xMinValue));
    },

    addTooltipCircles: function(chart, colorMap, dataObjs, dataObjsSelected, x, yAxis) {
        return chart.append("g")
            .classed("circles", true)
            .selectAll()
            .data(dataObjs).enter()
            .append("circle")
            .classed("tooltip", true)
            .classed("tooltip-circle", true)
            .attr("r", 5)
            .style("stroke", d => colorMap[d.category])
            .style("stroke-width", 3)
            .attr("cx", x)
            .attr("cy", (d, i) => yAxis[i](dataObjsSelected[d.category]))
            .attr("fill", "#FFF")
    },

    selectDataByDate: function(dataObjs, date) {
        const res = {};
        dataObjs.map(category => category.values
            .find(row => row.date.getTime() === date.getTime())
        ).forEach(obj => {
            res[obj.category] = obj.value;
        })
        return res;
    },

    addLine: function(xScale, yScale) {
        return d3.line()
            .x(function(d) {
                return xScale(d.date)
            })
            .y(function(d) {
                return yScale(d.value)
            })
    },

    addPath: function(chart, dataObjs, line, colorMap, appendG = true) {
        let g;
        if (appendG) {
            g = chart.append("g")
                .classed("trends", true)
                .classed("line-chart-element", true)
                .style("stroke-width", 3)
                .style("fill", "None");
        } else {
            g = chart.select("g.trends");
        }
        return g.selectAll()
            .data(dataObjs)
            .enter()
            .append("path")
            .attr("d", d => line(d.values))
            .attr("stroke", d => colorMap[d.category]);
    },

    // add legend text
    addLegendTextRight: function(chart, legendText, width, hasSecondAxis) {
        const xTran = hasSecondAxis ? 75 : 45;
        chart.select("g.legend")
            .selectAll()
            .data(legendText)
            .enter()
            .append("text")
            .attr("x", width + xTran)
            .attr("y", (d, i) => 112 + i * 25)
            .text(d => d)
            .attr("text-anchor", "left")
            .style("alignment-baseline", "middle")
            .style("font-size", 11)
    },

    // add legend for the plot
    addLegendRight: function(chart, width, keys, colorMap, legendText, hasSecondAxis, addLegendSymbol = null) {
        const colors = d3.scaleOrdinal()
            .domain(keys)
            .range(keys.map(d => colorMap[d]));

        // add legend symbol
        addLegendSymbol = this.isNotNull(addLegendSymbol) ? addLegendSymbol : function() {
            const xTran = hasSecondAxis ? 50 : 20;
            chart.append("g")
                .classed("legend", true)
                .selectAll()
                .data(keys)
                .enter()
                .append("rect")
                .classed("legend-rect", true)
                .attr("x", width + xTran)
                .attr("y", (d, i) => 100 + i * 25)
                .attr("width", 20)
                .attr("height", 20)
                .style("fill", d => colors(d));
        }

        addLegendSymbol();
        this.addLegendTextRight(chart, legendText, width, hasSecondAxis);
    },

    // add legend for nation comparison section
    addLegendBottom: function(chart, keys, colorMap, legendText, hasSecondAxis) {
        const width = +chart.attr("width").slice(0, -2);
        const height = +chart.attr("height").slice(0, -2);

        // add legend for the plot
        const colors = d3.scaleOrdinal()
            .domain(keys)
            .range(keys.map(d => colorMap[d]));

        // add legend symbol

        chart.append("g")
            .classed("legend", true)
            .selectAll()
            .data(keys)
            .enter()
            .each(function(d, i) {
                // add rect
                d3.select(this)
                    .append("rect")
                    .classed("legend-rect", true)
                    .attr("x", 20 + 85 * i)
                    .attr("y", 0)
                    .attr("width", 20)
                    .attr("height", 20)
                    .style("fill", d => colors(d));
                // add text
                d3.select(this)
                    .append("text")
                    .attr("x", 45 + 85 * i)
                    .attr("y", 14)
                    .text(legendText[i])
                    .attr("text-anchor", "start")
                    .style("font-size", 12.5);
            })
    },

    // update legend text
    updateLegendText: function(chart, width, keys, colorMap, legendText) {
        chart.selectAll("g.legend text")
            .data(legendText)
            .transition()
            .text(d => d);
    },

    // a numpy.linspace implementation for JS
    linspace: function(start, end, n) {
        const out = [];
        const delta = (end - start) / (n - 1);

        let i = 0;
        while (i < (n - 1)) {
            out.push(start + (i * delta));
            i++;
        }

        out.push(end);
        return out;
    },

    // sample in log space
    logspace: function(start, end, n) {
        start = Math.log10(start);
        end = Math.log10(end);
        const samples = this.linspace(start, end, n);
        return samples.map(x => 10 ** x);
    },

    // generate power math format
    formatPower: function(d) {
        const prefix = d < 0 ? "⁻" : "";
        const number = (d + "").split("").map(function(c) {
            return "⁰¹²³⁴⁵⁶⁷⁸⁹"[c];
        }).join("");
        return prefix + number;
    },

    // a function for adding color bar for world map
    addColorBar: function(svg, colorMap, valueMin, valueMax, scale, fill, height) {
        const samples = scale === "log" ? this.logspace(valueMin, valueMax, 100)
            : this.linspace(valueMin, valueMax, 100);
        const colors = samples.map(colorMap);

        // create colour scale
        const colorScale = d3.scaleLinear()
            .domain(samples)
            .range(colors);

        // style points
        d3.selectAll('circle')
            .attr('fill', function(d) {
                return colorScale(d.z);
            });

        // clear current legend
        svg.selectAll('*').remove();

        // append gradient bar
        const gradientId = `${svg.attr("id")}-gradient`;
        const gradient = svg.append('defs')
            .append('linearGradient')
            .attr('id', gradientId)
            .attr('x1', '0%') // bottom
            .attr('y1', '100%')
            .attr('x2', '0%') // to top
            .attr('y2', '0%')
            .attr('spreadMethod', 'pad');

        // values for legend
        const pct = this.linspace(0, 100, 100).map(function(d) {
            return Math.round(d) + '%';
        });

        d3.zip(pct, colors).forEach(function([pct, scale]) {
            gradient.append('stop')
                .attr('offset', pct)
                .attr('stop-color', scale)
                .attr('stop-opacity', 1);
        });

        const legendHeight = height - 20;
        const legendWidth = 20;

        svg.append('rect')
            .attr('x1', 0)
            .attr('y1', 10)
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('fill', `url(#${gradientId})`);

        // create a scale and axis for the legend
        let legendScale;
        if (scale === "log") {
            legendScale = d3.scaleLog()
                .domain([valueMin, valueMax])
                .range([legendHeight, 0])
        } else {
            legendScale = d3.scaleLinear()
                .domain([valueMin, valueMax])
                .range([legendHeight, 0])
        }

        let legendAxis = d3.axisRight()
            .scale(legendScale);

        legendAxis = scale !== "log" ? legendAxis : (fill === "confirmed" ? legendAxis.ticks(5) : legendAxis.ticks(4));

        legendAxis = legendAxis
            .tickFormat(d => {
                if (scale === "percentile") {
                    return (100 - d) + "%";
                } else if (scale === "log") {
                    return 10 + this.formatPower(Math.round(Math.log10(d)))
                } else {
                    if (fill === "confirmed") {
                        return +d / 1e6 + "M";
                    } else {
                        return d;
                    }
                }
            });

        svg.append("g")
            .style("font-size", scale === "log" ? 12 : null)
            .attr("class", "legend-axis")
            .attr("transform", "translate(" + legendWidth + ", 0)")
            .call(legendAxis);
    },

    // an Annotation class for better using d3-annotation library
    Annotation: class {
        constructor(label, title, x, y, dx, dy, color, wrap = null) {
            this.note = {label: label, title: title, wrap: wrap};
            this.x = x;
            this.y = y;
            this.dx = dx;
            this.dy = dy;
            this.color = color;
        }
    },

    // make annotation and remove annotation functions
    makeAnnotations: function(annotationObjs) {
        return d3.annotation().annotations(annotationObjs);
    },

    removeAnnotations: function(...selectors) {
        selectors.forEach(selector => {
            d3.select(selector).remove();
        });
    },

    // a function can transform the radian to angle degree
    radian2angle: function(radian) {
        return (180. / Math.PI) * radian
    },

    // remove given item from given array
    removeItem: function(arr, value) {
        const index = arr.indexOf(value);
        if (index > -1) {
            arr.splice(index, 1);
        }
        return arr;
    }
}
