# The Geography of Gratitude
## A natural language processing + cultural ecosystem services + geography project
### Done with about 14 days notice and only one prior programming class under my belt

The goal of this project was to investigate what Americans are thankful for on Thanksgiving, or at least what they say they are thankful for on Twitter.
The aim of this project was to answer several specific questions:


1. Is there a semantic difference in what Americans are grateful for based on the type of land they tweet from on Thanksgiving
    - based on the notion of cultural ecosystem services - that different environments influence peoples' values and perceptions of self, others, and society
2. Are there semantic differences in gratitude between states?
3. What do individual states disproportionately tweet about?


Looking back on it, this was implemented pretty poorly. However, this was also one of the single most valuable projects I have ever done in a school setting and I am proud of the creativity that it took to complete.
I'd like to revisit this at some point, but in the meantime I will leave my ideas here for people to check out.


You can see the final web app [here](https://uvm.maps.arcgis.com/apps/webappviewer/index.html?id=340a33fb64094eeaa64ec18bc25151c3)


The rest of this page will be the original `README.md` that I submitted with the assignment - just to give you an idea of how I went about doing this. Like I said, it was messy.


# Data/

### /cb_2017_us_county_20m/
- `cb_2017_us_county_20m`: Shapefile containing county polygons. Downloaded from https://www.census.gov/geo/maps-data/data/cbf/cbf_counties.html

### /CountyLevelData/
- `cb_2017_us_county_20m_Projec`: Same as cb_2017_us_county_20m except projected to the same coordinate system as the points
- `CountyLevelData`: The shapefile containing polygons for counties and county-level environmental/economic information. This file was created in ArcGIS from running the zonal statistics (majority) function on the NLCD 2011 land cover class raster using counties as zones, and running the zonal statistics (mean) function on the NLCD 2011 percent imperviousness raster (same zones).
- `es16.xls`: First, I sincerely apologize that this is an excel file. This is the county-level economic data that I joined to the counties based on the county and state IDs. The only two variables I ended up using was number of people in poverty and median household income. Downloaded from https://www.census.gov/data/datasets/2016/demo/saipe/2016-state-and-county.html


### /GeoOnly_US-labeled/
- `GeoOnly_US-labeled`: Contains tweets, their assigned topcs/topic words from LDA, and locations. This is derived from all geotagged tweets, but only contains those within the US. Written in `LDA_and_mapping.py, lines 165-204`.

### /GeoOnly_US/
- `GeoOnly_US`: Kinda goofed on my part, but this is another GIS-created dataset that has better spatial variables than the other one. Instead of running zonal statistics for the counties, which I realized was a pretty bad way to get good approximations for imperviousness or land cover, I ran the `extract multi-values to points` function. This assigns the raster value to the point that the point overlays, eliminating the need for zonal stats. I should've done this before, as it's much more computationally efficient as well.

### /CountyTweets/
- `CountyTweets`: Another shapefile, this one written in code (`LDA_and_mapping.py, lines 218-257`), not GIS. This is derived from GeoOnly_US, and is the final shapefile that I make/use. This contains every variable of interest and is what I used for my mapping web app and all subsequent language analyses.

# Scripts
- `TweetsUsingAPI.py`: The script I used to pull tweets from the streaming API on Thanksgiving. I let this run between 9 AM and 12:30 PM EST.

- `ProcessingTweets.py`: This file basically just contains useful functions for cleaning up the tweet json. Most of these are adapted from HW03. Pretty nifty stuff for twitter-specific data cleaning and formatting.

- `LDA_and_mapping.py`: The meat and potatoes, the bane of my existance, the reason for the season, whatever you want to call it. This is where I did 90% of the data cleaning/modelling. All the LDA and shapefile manipulation/analysis is in here. Also some plotting/mapping.

- `YuleCoefs.py`: Where I do all the Yule coefficient calculations. This code was written on Thursday, largely adapted from HW03 and also largely out of frustration with the black-boxness of LDA.

- `notes.pdf`: This is how I kept track of my thought process. This also contains the parameter I used for the last 7 LDA model trials, along with the top words for each topic. 
