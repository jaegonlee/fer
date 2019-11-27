import netP5.*;
import oscP5.*;
import processing.video.*;

Capture cam;
OscP5 oscP5;

int found;
int[] facePos;
String emotion;

void setup() {
  size(640, 360, P2D);
  oscP5 = new OscP5(this, 12345);
  oscP5.plug(this, "found", "/found");
  oscP5.plug(this, "face", "/face");
  oscP5.plug(this, "emotion", "/emotion");

  String[] cameras = Capture.list();

  println("Available cameras:");
  printArray(cameras);

  cam = new Capture(this, 640, 360);
  cam.start();
}

void draw() {
  if (cam.available() == true) {
    cam.read();
    image(cam, 0, 0);

    if (found != 0) {
      fill(255);
      text(emotion, facePos[0], facePos[1]-5);
      noFill();
      strokeWeight(2);
      stroke(255, 0, 0);
      rect(facePos[0], facePos[1], facePos[2], facePos[3]);
      drawFacePoints(); 
      drawFacePolygons();
    }
  }
}

void drawFacePoints() {
}

//--------------------------------------------
void drawFacePolygons() {
}

public void found(int i) {
  found = i;
}

public void face(int[] raw) {
  facePos = raw; 
}

public void emotion(String i) {
  emotion = i;
}
