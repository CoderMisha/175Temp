---
layout: default
title: Final Report
---
# {{page.title}}

## Video

## Project Summary
The goal of this project is to create a machine learning model to remove villagers from image screenshots taken in minecraft. We want to be able to input an image containing villagers into the model, and have it output an image with no villagers and the background reconstructed as if they were never there.

![Before After Eraser](assets/before_after_eraser.png)

Machine learning is very useful in this task because it would be very difficult to hard code a program for the infinite amount of villager positions and terrain. Additionally, after removing the villagers from the image, the program must generate a reconstruction of what the background behind the removed villager would look like. A task such as this is infeasible without the utilization of machine learning, since the program must learn how minecraft worlds are laid out and infer the details of the removed section from surrounding attributes and details in the input image. This model could be used to help create screenshots of landscapes in minecraft without having to manually remove villagers or mobs beforehand. It could also be extended to real life scenarios to remove people from landscape shots or remove tourists from personal pictures.
