# Setup of a Standardized Testbed for the Vision-based Furuta Pendulum

The instructions provided here can be used to build a Vision-based Furuta Pendulum hardware setup for reproducible and standardized hardware learning experiments based on a video input.

The setup consists of the Quanser Qube Servo 2, a high speed camera and a white box where both devices are mounted:
 

<p align="center" float="left">
  <img src="pictures/overview.jpg" height="200" />
  <img src="pictures/qube.jpg" height="200" />
</p>



## Quanser Qube Servo 2 

The *Qube-Servo 2* can be connected to the computer via USB, the Power LED should be green and the USB LED should be red. The driver will be installed automatically, the USB LED will turn to green if successful. If a driver is already installed it is recommended to uninstall it before connecting the device, some communication errors between the hardware and Python can be avoided.

The original power cables are not optimal as the pendulum may get caught in the plug when falling down. Especially when running long learning experiments on the hardware this may be annoying. An easy way around this is to get an angle plug from an electronics supply store and solder it onto the power cable (see [here](pictures/angle_plug.jpg)).

We also glued rubber bumpers on the Qube next to the cable connecting the motor with the qube. This protects the cable when high voltages are applied and serves as hardware limits for the arm angle (see [here](pictures/bumpers.jpg)).

## Camera Setup

We use a high speed camera for running vision-based experiments. The [Flir Blackfly S](https://www.flir.de/products/blackfly-s-usb3/) runs at a sample frequency of 522 Hz and thereby allows to run serial control cycles where a picture is taken and a control input calcualted afterwards.

The camera should be mounted on a tripod to not change the position of the camera. We added a flat LED light source for controlled light conditions (details see [picture](pictures/camera_setup.png)).

<p align="center" float="left">
  <img src="pictures/camera_front.jpg" height="300" />
  <img src="pictures/camera_side.jpg" height="300" /> 
</p>

## Environment

To standardize the environment for vision experiments we mounted the pendulum on a pedestal in a white box. We build the box in a carpentry with coated wood boards. The camera tripod was put in a wooden block to fix its position.

## Bill of Materials

The setup can be reproduced under a cost of 10.000 €, a BOM can be found [here](BOM.pdf).