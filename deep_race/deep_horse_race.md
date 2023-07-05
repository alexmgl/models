# Race Predictor

## Feature Engineering

**1. Horse Features**
   - Age
   - Compute if horse is placed
   - Last place of the horse
   - Last draw position
   - Average finish distance behind the leader (adjusted to race distance)
   - Rest time between races 
   - Diff weight between two races
   - Distance preference vs current length 
   - Race style (use a clustering algo to determine this)
   - Compute the horse ratings
   - Numbers of : runs, win, placed
   - Ratio of win and placed
   - Ratio of win and placed according to the venue
   - Current Handicap (kg) / Average handicap (kg)
   - Handicap weight / Horse weight or (handicap + jockey) / horse weight
   - Surface preference
   - Distance preference
   - Class preference
   - Travel distance to race (Is the horse out of country? Travel distance - maybe get google maps to calc distance?)
   - Win odds
   - Place odds
   - Speed or pace metric
   - Boolean maiden (a horse that never won a race)
   - Ranking (some kind of multi-agent ELO / ranking score)
   - Trend of the above ranking score 
   - Previous odds spread - Was there a significant difference between starting and closing odds? Odds risk - standard deviation, did this go up or down?
   - Average odds spread? Is this horse known for gaming the odds?

**2. External Features**
   - Venue (+ venue bias)
   - Surface
   - Race Class
   - Going
   - Temperature (°C)
   - Rain
   - Wind 


**3. Trainer Features**
   - Stable size
   - Win ratio
   - Place ratio
   - Average races per horse in stables
   - Average ranking of horse in stable
   - Odds informaiton - Does this trainer use inside information to trade? Big odds moves for the trainer? Any trend here?

**4. Jockey Features**
   - Win ratio
   - Place ratio
   - Odds informaiton - Does this trainer use inside information to trade? Big odds moves for the jockey? Any trend here?
   - Jockey weight, height (maybe bmi?)
   - Foreign jockey?
   - Average class of race vs current?


## Model Development

- 14 binary output variables. 
- Model required [14] horses as input. Fill dummy horse features with 0s if there are insufficient runners.
- If new horse without data, find the average data of a similar horse on the first race and use that.
