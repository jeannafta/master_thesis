// Defined by user
int analogPin0 = A0; // Pin where Vout is measured
int analogPin1 = A1; // Pin where Vout is measured
int Vin = 5; // Total tension
float R20 = 10000; // The known resistance
float R21 = 10000; // The known resistance
int dt = 1000; // Delay between measurements in ms

// Initialization of other variables
int raw0 = 0; // Raw reading from the analog pin
int raw1 = 0; // Raw reading from the analog pin
float Vout0 = 0; // Intermediate tension
float Vout1 = 0; // Intermediate tension
float R10 = 0; // Thermal resistance
float R11 = 0; // Thermal resistance

// Start
void setup() {
  Serial.begin(9600);
}

// Measure
void loop() {

  //  Read from the analogue pin, value from 0 to 1023
  raw0 = analogRead(analogPin0);
  raw1 = analogRead(analogPin1);

  if (raw0) {
    Vout0 = Vin * raw0 / 1023.0; // transform the analog output to intermediate tension
    R10 = R20 * (Vin/Vout0 - 1); // Apply Ohm's law to get resistance    
  }

    if (raw1) {
    Vout1 = Vin * raw1 / 1023.0; // transform the analog output to intermediate tension
    R11 = R21 * (Vin/Vout1 - 1); // Apply Ohm's law to get resistance
  }
  
  
  //Serial.print("R10 : ");
  //Serial.println(R10);   // add ln after print to go to 
  //Serial.print("R11 : ");
  //Serial.println(R11);
  //Serial.println("  ");
 
  Serial.print(R10);
  Serial.print(",");
  Serial.println(R11);

  //Serial.print("T0 : ");
  //Serial.print(",");
  //Serial.println(T0);
  //Serial.print("T1 : ");
  //Serial.print(",");
  //Serial.println(T1);

  delay(dt);

}