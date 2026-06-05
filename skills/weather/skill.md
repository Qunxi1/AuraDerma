# weather_check skill

## Purpose
Fetch real-time temperature and humidity for a user's city to provide climate-adapted skincare recommendations. Same season, different regions need different products (e.g., Guangdong summer = hot & humid → lightweight水乳; Yantai summer = cool & dry → cream).

## Inputs
- city: city name in Chinese or English

## Behavior
1. Call wttr.in API to fetch structured weather JSON.
2. Extract temperature (°C), humidity (%), and weather condition.
3. Return structured data.

## Output
```json
{
  "city": "广州",
  "temperature": 32,
  "humidity": 67,
  "condition": "Patchy rain nearby"
}
```

## Usage notes
- Call before product recommendation to adapt suggestions to local climate.
- Hot & humid → lightweight, oil-control, breathable products.
- Hot & dry → lightweight + hydration-focused.
- Cool & humid → balanced products.
- Cool & dry → richer, barrier-repair products.
