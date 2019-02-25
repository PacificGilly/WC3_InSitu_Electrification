// Oscillating Microbalance board OMB v.8.2.3
/*
  11th Sep 2017
  MWA Modified to report last value of PLL read in every line for use in less frequent reads.
  This prevents missing values when 0 is reported
  
  16th March 2017
  MWA v8.2.1 fixed with Full/Target run cyles of Max amp, PLL and FHT measurements for testing
  MWA v8.2.2 initiated for FHT led run method

  13th March 2017
  MWA v8.2 fixed as 'working version' with I2C to Pandora, but pre-FHT
  MWA v8.2.1 initiated for FHT testing

  FHT implementation test version
  MWA 13th Feb 2017

  I2C implementation functioning in this version
  MWA 1st Feb 2017

  LCD display version
  RGH 21st Sep 2016

  ver 7 to use Arduino as a programmer
  RGH 15th Sep 2016

  varies excitation frequency in a sweep from ~ 10Hz to 30 Hz
  measures response amplitude at each frequency

  MWA edits 08-11-2015
  Sweeps home in on measured resonant frequency
  LCD now outputs measured frequency (instead of driven), including estimates indicated with a '*'
  >>>Test in wind chamber with no drive to look for spontaneous resonance
  >>>Add calibration run (outside loop) to calculate equation for calculating and displaying mass
  >>>Next mods to measure uncertainty

  - output frequency generated on D13 (but pcb will use D0 in due course)
  -amplitude measured on A0 (U22 pin 23) (yellow)
  - drive on D13 (U22 pin2) (grey) NB pin19 on modified board
  - freq on D2 (U22 pin4) (white)
  - lock on D4 (U22 pin6) (blue)
  (and black wire GND to GND)
*/

#define LOG_OUT 1 // use the log output function
#define FHT_N 256 // set to 256 point fht

#include <FHT.h>
#include <Wire.h>
//#include <LiquidCrystal.h>
//LiquidCrystal lcd(12, 11, 10, 9, 8, 7);

String St;    // output string
int ampPin = A0;    // input pin for amplitude measurement
word fres0;    // resonant frequency
word fres;    // refined resonant frequency
int r;       // random offset
int freqPin = 2;  // input pin for frequency measurement
int lockPin = 4; // input pin for lock
int drivePin = 13; // output pin for drive
int state;  // lock pin state
word freqong; word ampmax; // onoing frequency and store variale for max amplitude frequency
word freq1; word freq2; word freq3; word maxfreq; int fsum;
float per1; float per2; float per3; float per4; float per5;
float per6; float per7; float per8; float per9; float per10;
word widthms; word Npulses; float halfperiodms; word amplitude;
word data[FHT_N]; word maxamp; float periodus; unsigned int padus;
int Sf; int fhtmax; int fhtbin; int fhtfreq; int range; int Nbins;

void setup() {
  // initialise digital pin 13 as an output
  pinMode(drivePin, OUTPUT); //drive
  // initialise digital pins 2 and 4 as inputs
  pinMode(freqPin, INPUT);  //freq
  pinMode(lockPin, INPUT); //lock

  Wire.begin(3);                // join i2c bus with address #3
  Wire.onRequest(requestEvent); // register event

  Serial.begin(9600);
  Serial.println("#OMBv8");
  Serial.println();

  //  lcd.begin(8, 1);
  //  lcd.setCursor( 0, 0 );   // left
  //  lcd.clear();
  //  lcd.print("OMBv8");
  //  delay(500);
  //  lcd.clear();

}


/////////////////////////////////////////////////////////////////////
// code associated

byte CountDigits(long Number) {
  // counts the number of digits in Number
  long t1;
  long t2;
  byte Ndigs;

  t2 = 1;
  t1 = 1;
  Ndigs = 0;

  while (t1 != 0) {
    Ndigs++;
    t2 = t2 * 10;
    t1 = Number / t2;
  }

  return Ndigs;
}

long TentoTheX(byte x) {
  long t;
  t = 1;
  for (byte i = 0; i < x; i++) {
    t = t * 10;
  }
  return t;
}

byte Digit(long Number, byte digitno) {
  // returns the digit counting the positions from right to left in a number
  // find power of ten for the digit sought, divide input number by that value, and then modulus the quotient with ten
  long t1;
  long t2;
  t1 = 1;
  t2 = 1;

  //for (byte i=1; i<digitno; i++){
  //t2=t2*10;
  //}
  t2 = TentoTheX(digitno) / 10;

  t1 = (Number / t2) % 10;

  return t1;
}

long TruncateNumber(long Number, byte Lengthdigs) {
  // code to shorten Number to contain the first (Lengthdigs) of digits
  long mult;
  byte Ndigits;
  byte Ndigs;
  long NumShort;
  byte RD;

  Ndigits = CountDigits(Number);
  //Serial.println(Ndigits);
  if (Lengthdigs >= Ndigits)
  {
    NumShort = Number;
    goto finish;
  }

  // extract the digits to round
  Ndigs = Ndigits - Lengthdigs;

  // compute 10^Lengthdigs - result is in mult
  //mult=1;
  //for (byte i=0; i<Ndigs; i++){
  //mult=mult*10;
  //}

  mult = TentoTheX(Ndigs);

  NumShort = Number / mult;

  // use the next most significant digit to possibly increase the value by 1 (ie rounding)
  RD = Digit(Number, Ndigs);
  if (RD >= 5)
  {
    NumShort = NumShort + 1;
  }

  //Serial.println(NumShort);
  //St=String(mult);

finish:
  return NumShort;
}

String DisplayIntegerAsFP(long Number, byte Npredigits, byte Ndecimals) {
  // splits up a long variable (Number) to give Ndigits before the decimal point and Ndecimals after
  // decimal point added and a string returned
  String DPright(10);
  String DPleft(10);
  String sign(1);
  String result;
  long dd1;
  long dd2;
  long tleft;
  long tright;
  long NumR;
  byte Ndigits;

  if (Number <= 0)
  {
    sign = "-";
  }
  else
  {
    sign = "";
  }

  Number = abs(Number);
  NumR = TruncateNumber(Number, Npredigits + Ndecimals);
  Ndigits = CountDigits(NumR);
  //Serial.println(NumR);

  // separate off the integer part left of the DP
  dd1 = Ndigits - Npredigits;

  //dd2=1;
  //for (byte i=0; i<dd1; i++){
  //dd2=dd2*10;
  //}

  dd2 = TentoTheX(dd1);

  //Serial.println(dd2);
  //Serial.println(" ");
  tleft = NumR / dd2;
  //Serial.println(tleft);
  DPleft = String(tleft);

  // and the decimal part
  tright = NumR - (tleft * dd2);
  tright = abs(tright);
  //Serial.println(tright);
  DPright = String(tright);

  // stick these bits together with a decimal point
  if (Ndecimals == 0)
  {
    result = sign + DPleft;
  }
  else
  {
    result = sign + DPleft + '.' + DPright;
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// OMB8 code

void freqout(word fout, int Npulses)    // generates a frequency fout (deciHz) for duration of Npulses
{
  float halfperiodms;
  float periodus;
  word widthms;
  unsigned int padus;

  halfperiodms = 5000.0 / (float)fout;
  widthms = int (halfperiodms);
  padus = int( 1000.0 * (float)(halfperiodms - widthms) );

  for (int j = 1; j <= Npulses; j++) {      //  generate pulse
    digitalWrite(13, HIGH);
    delay(widthms);
    delayMicroseconds(padus);

    digitalWrite(13, LOW);
    delay(widthms);
    delayMicroseconds(padus);
  };
};

void fhtSweep() {
  Sf = 100; // sampling frequency
  fhtmax = 0;
  for (int i = 0 ; i < FHT_N ; i++)
  { // save 256 samples
    delay(1000 / Sf);   // Sampling rate = 100 Hz (1000/100) = 10 ms delay
    int k = analogRead(ampPin);
    k -= 0x0200; // form into a signed int
    k <<= 6; // form into a 16b signed int
    fht_input[i] = k; // put real data into bins
  }
  fht_window(); // window the data for better frequency response
  fht_reorder(); // reorder the data before doing the fht
  fht_run(); // process the data in the fht
  fht_mag_log(); // take the output of the fht
  sei();
  for (int i = 12 ; i < (FHT_N / 2) - 13 ; i++) {
    if (fht_log_out[i] > fhtmax) {
      fhtmax = fht_log_out[i];
      fhtbin = i;
    }
  }
  range = 100 * (Sf / 2);
  fhtfreq = fhtbin * (range / 128);
  fhtfreq = fhtfreq / 10;

   // uncomment/comment following six lines to print/suppress FHT spectrum bins
      Serial.print("\n");
      for (int i = 0 ; i < FHT_N / 2 ; i++) {
        Serial.print(fht_log_out[i]); // send out the data (fht_log_out)
        Serial.print("\t");
      }
      Serial.print("\n");

}

void InitSweep(word f1, word f2, word fstep) {  // quick run to initialise for FHT sweep

  for (int f = f1; f <= f2; f = f + fstep) { // set the frequency (given in tenths of Hz, i.e. 200 is 20Hz)
    Npulses = 5;
    freqout(f, Npulses);
  };
  delay(500);
};

void Sweep(word f1, word f2, word fstep) {            // sweeps frequency from f1 to f2 in steps of fstep, and returns resonant frequency fres

  maxamp = 0;
  fres = 0;
  fsum = 0;
  for (int f = f1; f <= f2; f = f + fstep) { // set the frequency (given in tenths of Hz, i.e. 200 is 20Hz)
    halfperiodms = 5000.0 / (float)f;
    widthms = int (halfperiodms);
    padus = int( 1000.0 * (float)(halfperiodms - widthms) );

    delay(200);
    //Npulses= 500/widthms;                  // 1s duration each time
    Npulses = 10;
    amplitude = 0;
    freqout(f, Npulses);
    delayMicroseconds(300);                     // wait for drive to cease
    amplitude = amplitude + analogRead(ampPin);   // measure
    delay(widthms);
    amplitude = amplitude + analogRead(ampPin);   // see if any ringing remains after half cycle

    if (amplitude > maxamp) {                     // store
      maxamp = amplitude;
      fres0 = f;
    };

    freq1 = 0;
    state = digitalRead(lockPin);

    if (state == LOW || HIGH) { // reads five periods of signal wave if PLL shows 'lock' status
      per1 = pulseIn(freqPin, HIGH); // read half period in us
      per2 = pulseIn(freqPin, LOW);
      per3 = pulseIn(freqPin, HIGH);
      per4 = pulseIn(freqPin, LOW);
      per5 = pulseIn(freqPin, HIGH);
      per6 = pulseIn(freqPin, LOW);
      per7 = pulseIn(freqPin, HIGH);
      per8 = pulseIn(freqPin, LOW);
      per9 = pulseIn(freqPin, HIGH);
      per10 = pulseIn(freqPin, LOW);
      per1 = (per1 + per2 + per3 + per4 + per5 + per6 + per7 + per8 + per9 + per10) / 5; // mean period of oscillation
      freq1 = (1 / per1) * 1e7; // convert to dHz
      freqong = freq1; // store ongoing frequency value prior to filter in next line
      if (freq1 > f + 5 || freq1 < f - 5) { // if measured frequency outside reasonable range, discard from this variable, freq1
        freq1 = 0;
      }
      else if (freq1 <= f + 5 || freq1 >= f - 5) { // keep latest figure of freq1 in stored variable freq2
        freq2 = freq1;
      }
    };

    fsum = fsum + freq1;

    Serial.print(f);
    Serial.print("\t");
    Serial.print(amplitude);
    Serial.print("\t");
    Serial.print(freq1); // freq1(locked) or freqong (all freq locked or unlocked)
    Serial.print("\t");
    Serial.println();

    //    lcd.setCursor( 0, 0 );   // left
    //    lcd.clear();
    //    lcd.print(f);
    //    lcd.print(" ");
    //    //lcd.print(amplitude);
    //    lcd.print(freq1);

  }
  digitalWrite(13, LOW); // kill drive
}

// function that executes whenever data is requested by master
// this function is registered as an event, see setup()
void requestEvent() {
  Wire.write(highByte(amplitude));        // data bytes
  Wire.write(lowByte(amplitude));
  Wire.write(highByte(ampmax));
  Wire.write(lowByte(ampmax));
  Wire.write(highByte(freq2));          //freqong for all measured frequencies, freq1 for locked frequencies (every lock, not held)
  Wire.write(lowByte(freq2));
  Wire.write(highByte(fhtfreq));            // freq2 for updated (and held) every lock, freq3 for updated every sweep (like maxfreq), fhtfreq for fht
  Wire.write(lowByte(fhtfreq));
}

void loop() {

  InitSweep(200, 400, 50); // quick sweep from 20-40 Hz to provide vibration for FHT
  fhtSweep(); // run FHT function to find peak frequency (+-0.2 Hz)
  Serial.print("FHT frequency centre: ");
  Serial.println(fhtfreq);

  // freq2 = 0;
  freq3 = 0;
  maxfreq = 0;

  Serial.println("Drv\tAmp\tPLL");
  Serial.println("---\t---\t---");

  //    lcd.setCursor( 0, 0 );   // left
  //    lcd.clear();

  Sweep(fhtfreq - 4, fhtfreq + 4, 1); // target sweep from FHT freq -0.4 to FHT freq +0.4

  Serial.println();
  Serial.print("#Max amp. at ");
  Serial.println(fres0);     // print frequency corresponding to maximum amplitude
  maxfreq = fres0;
  ampmax = maxfreq;

  Serial.print("#PLL lock at ");
  Serial.print(freq2); // report last known value of frequency at PLL lock
  freq3 = freq2;
  Serial.println();
  Serial.println();

  //  St = DisplayIntegerAsFP(freq2, 2, 1);
  //    St = St + " Hz";
  //
  //    lcd.setCursor( 0, 0 );   // left
  //    lcd.clear();
  //    lcd.print(St);
  //    delay(200);
};


