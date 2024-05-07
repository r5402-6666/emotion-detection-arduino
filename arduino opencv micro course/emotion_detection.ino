#include <FastLED.h>

#define LED_PIN     6
#define LED_COUNT   18
#define LED_TYPE    WS2812B

CRGB leds[LED_COUNT];

void setup() {
  FastLED.addLeds<LED_TYPE, LED_PIN>(leds, LED_COUNT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char emotion = Serial.read();
    if (emotion == 'H') {
      // Happy emotion
      // Define LED pattern for happy
      fill_solid(leds, LED_COUNT, CRGB::Green);
      FastLED.show();
    } else if (emotion == 'S') {
      // Sad emotion
      // Define LED pattern for sad
      fill_solid(leds, LED_COUNT, CRGB::Red);
      FastLED.show();
    }
    // Define patterns for other emotions
  }
}
