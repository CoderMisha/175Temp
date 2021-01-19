---
layout: default
title:  Proposal
---

## Summary
Our project’s goal is to create an AI that can navigate a 3 dimensional parkour course from start to end in an efficient and quick manner. The input will be a set of blocks in a small radius around the agent that are part of a completable parkour course, a start and end position for the AI, and the number of blocks the AI has. The output will be a sequence of actions the AI makes to get from start to end. The AI will have the ability to walk, sprint, and jump as well as place blocks and water from a bucket to assist its traversal of the course. The course itself will be composed of solid blocks, iron bars, and ladders, along with lava and bottomless pits providing added danger.

## AI/ML Algorithms
We anticipate using reinforcement learning with neural function approximators, and are possibly considering also using some computer vision elements.

## Evaluation Plan
  Our metrics for analyzing the performance of our agent will take into consideration how far the agent travels in relation to the position of the end goal, how fast the agent is traveling through the map, how much damage the agent takes on its run of the course, and whether the agent survives to reach the end. As the agent progressively makes more and more attempts at the course, we expect its evaluation criterion score to increase until it finds and completes the optimal path consistently. The baseline case we will be testing against an agent that performs actions randomly, and we expect to see considerable difference in their performance scores. Possibly, we will compare the agent against ourselves as we attempt the courses as well. We will evaluate the two models according to our stated metrics on a small set of parkour courses that vary in size and difficulty.

  Our sanity case would be if there is a plain line of blocks to our goal, our agent can just run through in a line, and it would not choose other paths. We are also considering testing a 2D grid of blocks and lava to make sure our agent reasons and behaves properly before it is required to act in three dimensions. Visualization can be achieved by drawing out the agent’s path and color-coordinating the path color by its evaluation score for that path. Our moonshot case would be that the agent would be able to jump through a series of blocks and save itself by placing blocks or the water bucket if the distance between two blocks is too far, or create its own path to the goal utilizing the blocks it is given.
