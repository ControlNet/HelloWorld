install.packages("shiny")
install.packages("leaflet")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("RColorBrewer")
install.packages("shinyWidgets")
install.packages("ggdark")
library(shiny)
library(leaflet)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(shinyWidgets)
library(ggdark)

# read data
data.2019 <- read.csv("Pedestrian_Counting_System_2019.csv")
data.location <- read.csv("Pedestrian_Counting_System_-_Sensor_Locations.csv")

# assign sensor names
data.2019.name <- data.2019$Sensor_Name %>% unique
data.location.name <- data.location$sensor_name %>% unique

# check the non-matched data
data.2019.name.nonmatched <- data.2019.name[!data.2019.name %in% data.location.name]
data.location.name.nonmatched <- data.location.name[!data.location.name %in% data.2019.name]

# manually correction
data.location[data.location$sensor_name == "Melbourne Central-Elizabeth St (East)Melbourne Central-Elizabeth St (East)",]$sensor_name <- "Melbourne Central-Elizabeth St (East)"
name.map <- list(
  "Swanston St - RMIT Building 80" = "Building 80 RMIT",
  "Swanston St - RMIT Building 14" = "RMIT Building 14",
  "Collins St (North)" = "Collins Street (North)",
  "Flinders la - Swanston St (West) Temp" = "Flinders St-Swanston St (West)",
  "Flinders La - Swanston St (West) Temp" = "Flinders St-Swanston St (West)",
  "Lincoln-Swanston(West)" = "Lincoln-Swanston (West)",
  "Pelham St (S)" = "Pelham St (South)"
)

apply.name.map <- function(name) {
  if (name %in% names(name.map)) name.map[name]
  else name
}
data.2019 <- data.2019 %>% mutate(Sensor_Name = sapply(Sensor_Name, apply.name.map))

# grouped by the sensor names and years for circle marker radius
data.2019.year <- data.2019 %>%
  group_by(Sensor_Name) %>%
  summarise(avg_hourly_count = mean(Hourly_Counts))

# grouped by the sensor names and days for line chart
data.2019.day <- data.2019 %>%
  group_by(Day, Time, Sensor_Name) %>%
  summarise(avg_hourly_count = mean(Hourly_Counts))

# join the traffic data with location data by the sensor name
data.2019.year.merge <- merge(x = data.location, y = data.2019.year,
                              by.x = "sensor_name", by.y = "Sensor_Name")

data.2019.day.merge <- merge(x = data.location, y = data.2019.day,
                             by.x = "sensor_name", by.y = "Sensor_Name")

# select unique sensor names for select input
sensors <- data.2019.day.merge$sensor_name %>% unique
# day orders
day.order <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


ui <- fillPage(
  title = "Pedestrian Visualization",
  # set background
  setBackgroundColor("black"),
  # set headers with the title
  headerPanel(HTML("<h1 style=\"color: white;\">Pedestrian Visualization</h1>")),
  # main panel with split layout, containing two output
  mainPanel(splitLayout(
    leafletOutput("map"),
    plotOutput("plot")
  )),
  # the select input to
  pickerInput("sensor", "Sensor:", choices = sensors, selected = sensors[1],
              options = list(style = "btn-primary"))
)

server <- function(input, output, session) {
  # a function for mapping numeric values to colors
  cpal <- reactive({
    colorNumeric("RdYlGn", data.2019.year$avg_hourly_count)
  })

  # leaflet map output
  output$map <- renderLeaflet(
    leaflet(data.2019.year.merge) %>%
      addProviderTiles("CartoDB.DarkMatter") %>%
      addCircleMarkers(~longitude, ~latitude, layerId = ~sensor_name, color = ~cpal()(avg_hourly_count), stroke = FALSE,
                       fillOpacity = 0.9, label = ~as.character(sensor_name), radius = ~avg_hourly_count/150)
  )
  # update the select bar when the circle marker is clicked
  observe({
    updatePickerInput(session, "sensor", selected = input$map_marker_click$id)
  })

  output$plot <- renderPlot({
    data.2019.day %>%
      filter(Sensor_Name == input$sensor) %>%  # filter the sensor name selected
      mutate(Day = factor(Day, levels = day.order)) %>%  # reorder the days
      ggplot + # plot ggplot
      geom_line(aes(x = Time, y = avg_hourly_count, color = Day), size = 1.2) +
      facet_wrap(~Day, nrow = 1) +
      scale_color_discrete(breaks = day.order) + # show days in matched orders
      labs(title = input$sensor, y = "Average Hourly Count") + # set y-label and title for sensor
      lims(x = c(0, 23), y = c(0, 5000)) + # set range of x and y axis
      dark_theme_minimal() + # set black backgrounds
      theme(plot.background = element_rect(fill = "black", color = "black", size = 0)) # set black edge
  })
}

shinyApp(ui, server)
