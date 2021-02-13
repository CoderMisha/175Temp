---
layout: default
title:  Proposal
---

# {{page.title}}

## Summary
  The purpose of the project is to create an algorithm to remove villagers and other mobs from images in minecraft. Our machine learning model will be trained by feeding it an image containing villagers and other creatures matched with an image with all mobs removed, and it will be tasked with returning a new output image with the mobs erased and background filled in as if they were never there. This algorithm could be used to help players take screenshots of landscapes in minecraft without having to manually remove mobs beforehand, as well as extend to real life scenarios to remove people like tourists from pictures.

## AI/ML Algorithms
  We plan to use a convolutional neural network as a baseline model, and move on to a conditional generative adversarial network with modified U-Net generator and PatchGAN discriminator as our final model for our project.

## Evaluation Plan
  We will be evaluating the performance of our learner on data we manually collect of images with and without mobs through several different quantitative measures. Mainly, for our metrics we will use binary accuracy between image pixels and mean squared error between the expected and reconstructed images to measure the effectiveness of our models performance. The model will be trained on 75% of the data and then tested on 25% of the data to determine its overall performance. Our baseline model will be a convolutional autoencoder, and we will compare the effectiveness of the CAE to our final GAN model, which we expect will significantly improve our predicted image quality and accuracy. Our data will be collected through Minecraft screenshots, utilizing Malmo to quickly move the agent around to take different screenshots and add and remove mobs.

  Our first sanity case would be training the model on a single image pair and making sure the model’s output on the image matches the expected image very closely, ensuring the model actually learns a good latent space representation and learns how to reconstruct the image effectively. Another sanity case would be to have a small number of inputs of a villager against a completely solid-colored wall and have the model be able to remove the villager. Qualitatively, we can look at the output images manually and determine how well the algorithm has removed the mobs from the image and recreated the background. If we are able to, we would like to extend our algorithm to also be able to add mobs back to images, or possibly be able to remove mobs from a video.
  
## Appointment
4:30pm - 4:45pm, Friday, January 22, 2021
 
## Team Meetings
Monday and Friday at 4:00pm
