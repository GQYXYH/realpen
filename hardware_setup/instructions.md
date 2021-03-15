# Setup of a Standardized Testbed for the Vision-based Furuta Pendulum

## Quanser Qube Servo 2 

The *Qube-Servo 2* can be connected to the computer via USB, the Power LED should be green and the USB LED should be red. The driver will be installed automatically, the USB LED will turn to green if successful. If a driver is already installed it is recommended to uninstall it before connecting the device, some communication errors between the hardware and Python can be avoided.

The original power cables are not optimal as the pendulum may get caught in the plug when falling down. Especially when running long learning experiments on the hardware this may be annoying. An easy way around this is to get an angle plug from an electronics supply store and solder it onto the power cable.

We also glued rubb er bumpers on the Qube next to the cable connecting the motor with the qube. This protects the cable when high voltages are applied and serves as hardware limits for the arm angle (details see HERE).

## Camera Setup

We use a high speed camera for running vision-based experiments. The [Flir Blackfly S](https://www.flir.de/products/blackfly-s-usb3/) runs at a sample frequency of 522 Hz and thereby allows to run serial control cycles where a picture is taken and a control input calcualted afterwards.

The camera should be mounted on a tripod to not change the position of the camera. We added a flat LED light source for controlled light conditions.

## Environment

To standardize the environment for vision experiments we mounted the pendulum at a fixed position in a white box (see pictures below). We also attached the camera and light source on a tripod in the box.

## Bill of Materials

The setup can be reproduced under a cost of 10.000 €, a BOM can be found [here](BOM.pdf).