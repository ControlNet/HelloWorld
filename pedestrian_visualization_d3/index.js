const avgWidth = 960;
const avgHeight = 540;

const margin = {top: 10, right: 100, bottom: 30, left: 50};
const width = avgWidth - margin.left - margin.right;
const height = avgHeight - margin.top - margin.bottom;

const svg = d3.select("#svg-canvas")
    .attr("height", avgHeight)
    .attr("width", avgWidth);

const lineChart = svg
    .append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);

d3.csv("PedestrianCountingSystemWeekdaysAndWeekends.csv", data => {

    const days = ["Weekday", "Weekend"];

    // choose 2 color to represent weekday and weekend data
    const colorMap = {Weekday: "#82AAFF", Weekend: "#F78C6C"};

    // x axis
    const x = d3.scaleLinear()
        .domain([0, 24])
        .range([0, width]);
    lineChart.append("g")
        .attr("transform", `translate(0, ${height})`)
        .call(d3.axisBottom(x));

    // y axis
    const y = d3.scaleLinear()
        .domain([0, Math.max(...data.map(row => +row["HourlyAverage"]))])
        .range([height, 0]);

    lineChart.append("g")
        .call(d3.axisLeft(y));

    // Reformat the data as an array of {x, y} tuples
    const dataObjs = days.map(day => {
        return {
            day: day,
            values: data.filter(row => row["Day"] === day)
                .map(row => {
                    return {day: day, time: +row["TimeOfDay"], value: +row["HourlyAverage"]};
                })
        };
    });

    // Add the lines
    const line = d3.line()
        .x(function(d) {
            return x(d.time)
        })
        .y(function(d) {
            return y(d.value)
        })

    // prepare tooltip
    const tooltip = svg.append("g")
        .attr("class", "tooltip")
        .style("opacity", 0);

    const tooltipRect = tooltip.append("rect")
        .attr("class", "tooltip-rect")
        .attr("height", 50)
        .attr("width", 130)
        .style("fill", "white")
        .style("stroke", "black")
        .attr("rx", "5px")
        .attr("ry", "5px");

    const tooltipText = tooltip.append("text")
        .attr("class", "tooltip-text")
        .style("font-size", 12);

    tooltipText.append("tspan").attr("class", "tooltip-text-row1");
    tooltipText.append("tspan").attr("class", "tooltip-text-row2");
    tooltipText.append("tspan").attr("class", "tooltip-text-row3");

    // add annotations
    class Annotation {
        constructor(label, title, x, y, dx, dy, wrap = null) {
            this.note = {label: label, title: title, wrap: wrap};
            this.x = x;
            this.y = y;
            this.dx = dx;
            this.dy = dy;
            this.color = "#F07178"
        }
    }

    const annotations = [
        new Annotation("Pedestrian counts during weekdays fluctuate at peak hours, whereas counts during " +
            "weekends grow and shrink steadily throughout the day.","Morning Peak", x(8), y(992), -50, 15, 200),
        new Annotation("", "Lunchtime Peak", x(13), y(1234), -100, -50, 200),
        new Annotation("", "Evening Peak", x(17), y(1499), 100, 50, 200)
    ]

    const makeAnnotations = d3.annotation()
        .annotations(annotations)
    lineChart.append("g")
        .call(makeAnnotations)


    // for each line, weekend and weekday
    lineChart.append("g")
        .attr("id", "trends")
        .attr("class", "line-chart-element")
        .style("stroke-width", 3)
        .style("fill", "None")
        .selectAll()
        .data(dataObjs)
        .enter()
        .append("path")
        .attr("d", d => line(d.values))
        .attr("stroke", d => colorMap[d.day]);

    // plot circle for each data point
    // for all circles
    lineChart.append("g")
        .attr("id", "scatters")
        .attr("class", "line-chart-element")
        .style("stroke-width", 3)
        .selectAll()
        // for 2 groups: weekday and weekend
        .data(dataObjs)
        .enter()
        .append("g")
        .style("stroke", d => colorMap[d.day])
        .style("fill", "white")
        .selectAll()
        // for all points
        .data(d => d.values)
        .enter()
        .append("circle")
        .attr("r", 7)
        .attr("cx", d => x(d.time))
        .attr("cy", d => y(d.value))
        .on("mouseover", function() {
            d3.select(this)
                .style("stroke", "black");
            tooltip
                .style("opacity", 1);
        })
        .on("mousemove", function(d) {
            const mousePos = {x: d3.mouse(this)[0], y: d3.mouse(this)[1]}
            tooltipRect
                .attr("x", (mousePos.x + 70.) + "px")
                .attr("y", mousePos.y + "px");
            tooltipText
                .attr("text-anchor", "left");
            [`Day: ${d.day}`, `Hour: ${d.time}`, `Avg.Pedestrian: ${Math.round(d.value)}`]
                .forEach(function(line, i) {
                    tooltipText
                        .select(`.tooltip-text-row${i + 1}`)
                        .text(line)
                        .attr("x", (mousePos.x + 75.) + "px")
                        .attr("y", (15. + mousePos.y + 15 * i) + "px")
                        .attr("text-anchor", "left");
                })
        })
        .on("mouseout", function(d) {
            d3.select(this)
                .style("stroke", colorMap[d.day])
            tooltip
                .style("opacity", 0);
            tooltipRect
                .attr("x", -10000).attr("y", -10000);
            tooltipText
                .selectAll("tspan")
                .text("");
        });
})

