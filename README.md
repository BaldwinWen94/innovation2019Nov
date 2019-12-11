# innovation2019Nov
This project serves as the waste classification part of Bumble Bee, which is 1st place of (Paypal Shanghai) Innovation Day 2019.

Use Pytorch & RaspberryPi & Bootstrap4

----

### Build Classify Model
Build pytorch model for waste classification.
GPU with CUDA recommended, I spent 3 hours training 25,000 images with CUDA of GTX2070.


### Classify Server
Provide api to classify a single image with pre-trained model


### RaspberryPi Server
Provide api on RaspberryPi to call the classify server, thus enables the UbiquityRobotics car to call classify server.


### Web
Webpages for road show.
