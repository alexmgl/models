# Race Predictor

## Feature Engineering

**1. Horse Features**
   - Age
   - Compute if horse is placed
   - Last place of the horse
   - Last draw position
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
   - Travel distance to race
   - Win odds
   - Place odds
   - Speed or pace metric 


**2. External Features**
   - Venue (+ venue bias)
   - Surface
   - Race Class
   - Going
   - Temperature (°C)
   - Rain
   - Wind 


**3. Trainer Features**

**4. Jockey Features**


## Model Development

- 14 binary output variables. 
- Model required [14] horses as input. Fill dummy horse features with 0s if there are insufficient runners.
- If new horse without data, find the average data of a similar horse on the first race and use that.