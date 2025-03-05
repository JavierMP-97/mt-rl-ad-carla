# Deep Reinforcement Learning for Autonomous Driving in Urban Environments

This repository is my **master's thesis**, a spiritual sequel to my bachelor’s thesis, where I first attempted to teach an autonomous agent how to drive using Deep Reinforcement Learning (DRL). Back then, the task was relatively simple: follow a road in a clean, noise-free environment using Soft Actor-Critic (SAC) and an autoencoder for feature extraction.

This time, I raised the difficulty level. In my master’s thesis, I set out to solve a more complex driving task—navigating urban environments filled with intersections, obstacles, and noisy input data. If my first project was like teaching a toddler to walk, this was like preparing them for the chaos of rush-hour traffic.

## Watch the Project in Action  

https://github.com/user-attachments/assets/c0f33dfe-c8d0-4b22-a656-5d1a427855c5

## Key Improvements Over the Previous Project

**From Simple Roads to City Streets**: The agent now faces a real challenge—urban navigation, where intersections and unexpected situations make decision-making much harder.

**Following Directions at Intersections**: Unlike before, the agent isn’t just blindly following a road. Now, it can take turns at intersections based on provided navigation instructions.

**Semantic Segmentation for Feature Extraction**: Instead of relying on raw pixel data, the autoencoder now learns high-level features from semantic segmentation, helping the agent distinguish roads, sidewalks, and lane markings—because let’s face it, knowing the difference between a road and a wall is pretty useful.

**More Meaningful Evaluation Metrics**: This time, we go beyond just watching the agent drive (or crash). The models are evaluated based on:
- Reward – A carefully crafted scoring system that encourages smooth, safe, and efficient driving.
- Time Between Failures – How long the agent can drive before making a critical mistake.
- Percentage of Autonomous Driving – The proportion of time the agent successfully drives without intervention.

## The Challenge
Urban driving is a whole new beast. The agent doesn’t just need to stay on the road—it needs to navigate through intersections, react to unpredictable scenarios, and avoid making questionable driving decisions. While this project made significant progress, there’s still plenty of room for improvement (as always).

## Publications  

**Master’s Thesis**: [Semantic Segmentation for Autonomous Driving using Reinforcement Learning](https://hdl.handle.net/10016/37956)

**Bachelor’s Thesis**: [Deep Reinforcement Learning for Autonomous Driving](https://hdl.handle.net/10016/30350)

**[Bachelor’s Thesis GitHub](https://github.com/JavierMP-97/bt-rl-ad-robot/)**. My first attempt at teaching a car to drive using Deep Reinforcement Learning (DRL)—featuring Soft Actor-Critic (SAC), an autoencoder for feature extraction, and enough spaghetti code to make an Italian grandmother weep

**[CARLA_RL](https://github.com/JavierMP-97/carla_rl)** This is a much more refined version, designed to help students quickly start their autonomous driving projects. It provides a CARLA environment with data collection, rewards, feature extraction, and a complete pipeline to train a baseline agent.
