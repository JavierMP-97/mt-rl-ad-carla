# Deep Reinforcement Learning for Autonomous Driving in Urban Environments

This repository is my **master's thesis**, a spiritual sequel to my bachelor’s thesis, where I first attempted to teach an autonomous agent how to drive using Deep Reinforcement Learning (DRL). Back then, the task was relatively simple: follow a road in a clean, noise-free environment using Soft Actor-Critic (SAC) and an autoencoder for feature extraction.

This time, I raised the difficulty level. In my master’s thesis, I set out to solve a more complex driving task—navigating urban environments filled with intersections, and noisy input data.

In this project, I build on the previous work by training an agent to follow directions while incorporating semantic segmentation into the encoder for additional contextual understanding. Finally, I evaluate the impact of encoding semantic segmentation compared to using raw images, analyzing its effectiveness in improving driving performance.

While my development skills improved significantly during this project, there is still plenty of room for growth (as always). However, this work showcases a more methodical and refined approach to research projects.

## Overview

**From Simple Roads to City Streets**: The agent now faces a real challenge—urban navigation, where intersections and unexpected situations make decision-making much harder.

**Following Directions at Intersections**: Unlike before, the agent isn’t just blindly following a road. Now, it can take turns at intersections based on provided navigation instructions.

**Semantic Segmentation for Feature Extraction**: Instead of relying on raw pixel data, the autoencoder now learns high-level features from semantic segmentation, helping the agent distinguish roads, sidewalks, and lane markings—because let’s face it, knowing the difference between a road and a wall is pretty useful.

**Improved VAE loss**: Designed a custom training loss metric that focuses on improving attention and performance where it matters.

**More Meaningful Evaluation Metrics**: This time, we go beyond just watching the agent drive (or crash). The models are evaluated based on:
- Reward – A carefully crafted scoring system that encourages smooth, safe, and efficient driving.
- Time Between Failures – How long the agent can drive before making a critical mistake.
- Percentage of Autonomous Driving – The proportion of time the agent successfully drives without intervention.

## Watch the Project in Action  

https://github.com/user-attachments/assets/c0f33dfe-c8d0-4b22-a656-5d1a427855c5

## Summary

### Problem Definition

The objective is to train an agent to follow directions using front-facing camera images in a realistic 3D urban and highway environment. Key aspects of the problem include:
- **Reinforcement Learning**: The agent learns a driving policy using a reinforcement learning algorithm.
- **Perception System**: A single front-facing camera captures the environment. Features are extracted using an Autoencoder trained on Semantic Segmentation.
- **Follow Directions**: The agent should respond to navigation instructions to reach specific destinations.
- **Vehicle Dynamics**: Obeys real-world physics (inertia, traction, acceleration constraints), requiring indirect speed control via acceleration and braking.
- **Complex Environment**: Includes roads, buildings, vegetation, traffic signs, varying lighting, weather, and road conditions, introduce noise in perception and control.
- **Research Question**: How does encoding semantic segmentation affect the agent’s ability to learn to drive?

### Model Design

#### VAE Encoding

Images are preprocessed using a VAE trained to reconstruct Semantic Segmentations:
- **Input**: Images of shape [128, 256, 3].
- **Convolutional Layers**: 4 Layers with [32, 64, 128, 256] filters, kernel size 5x5 and stride 2.
- **Output**: Latent vector of size 128.
- **Semantic Segmentation Classes**: 3 classes representing the road, road lines, and everything else.
- **Optimized Training Loss**: Road lines, though occupying a small portion of the image, are a crucial feature. To account for this, the loss for road line pixels is assigned a significantly higher weight than the other classes.

![Screenshot 2025-03-08 120745](https://github.com/user-attachments/assets/74840c95-d3b7-4979-bb85-7ab6e7a472bc)

#### State Representation

- 128-dimensional latent vector
- Last N actions performed (steering and acceleration), totaling N*2 values.
- Instruction for next intersection (turn left, right, or follow the road)
- Speed
- IMU Data (linear and rotational forces)

#### Action Space

Continuous action space controlling two dimensions:
- Steering angle
- Acceleration

#### Reward Model

If the vehicle crashes or exits the road (terminating the episode):
- \- 10 - speed * 0.25

If the vehicle doesn't move for more than 2 seconds:
- 0

Else:
- \+ Acceleration
- \+ 1 - (cte / $cte_{max}$(2 m))
- \+ 1 - (angle_diff / $angle\_diff_{max}$(90º))

*angle_diff is the difference in angle between the direction of the vehicle and the direction of the road

This reward encourages movement and speed, alignment with the road (avoiding zigzagging movements), while still being cautious while turning.

#### Agent Architecture

The algorithm chosen was Soft Actor Critic, given the great results that it offered in me Bachelor's Thesis.

The agent uses the following architecture:
- VAE encoding as input
- 2 hidden layers with 1024 and 512 neurons and ELU activation function, shared by actor and critic 
- 2 output neurons for actor
- 1 output neuron for critic

### Experimental Results

- The VAE generalized better than a traditional autoencoder

![VAE o Autoencoder Tradicional2_rec](https://github.com/user-attachments/assets/a37f76fb-1646-412f-8d22-1de7b0c352ff)

- The agent needs information on the forces that the vehicle is suffering, either by including IMU data or giving it information from the previous actions.

![Sin IMU  Ultimas Acciones_ma_reward](https://github.com/user-attachments/assets/2013a096-19f1-4933-8fd1-a13866ced568)

- Encoding Semantic Segmentation in the VAE was fundamental to solving this task.

![Entrenamiento del Modelo Final_ma_reward](https://github.com/user-attachments/assets/c79efd2b-3680-4d63-8f19-accc72e96250)

Test results:

| Encoder                      | Average Reward  | Average Time per Interruption | Autonomy Percentage (6s penalty per interruption) |
|------------------------------|----------------|------------------------------|----------------------|
| With Semantic Segmentation   | 2814.7619      | 165.03                        | 96.49                |
| Without Semantic Segmentation | 631.6270       | 21.6                          | 78.26                |

### Conclusion

This project demonstrated that using Semantic Segmentation as the target for a Variational Autoencoder (VAE) significantly improves reinforcement learning performance in complex autonomous driving tasks. By training an autoencoder to reconstruct semantic segmentation instead of raw images, the agent learned to drive more effectively, even in visually and structurally complex environments. Without this technique, as the experiments showed, achieving robust navigation in urban environments would have been far more challenging.

This project was a significant step forward in my journey with Machine Learning for autonomous driving. Compared to my Bachelor’s thesis, this work represents a more structured, methodical, and technically advanced approach to solving complex reinforcement learning challenges and research tasks.

My work in autonomous driving didn't stop here. I kept helping other students with their own theses and projects. For this purpose I've developed a small toolkit to quickstart your autonomous driving projects with **[CARLA_RL](https://github.com/JavierMP-97/carla_rl)**.

## Publications  

**Master’s Thesis**: [Semantic Segmentation for Autonomous Driving using Reinforcement Learning](https://hdl.handle.net/10016/37956)

**Bachelor’s Thesis**: [Deep Reinforcement Learning for Autonomous Driving](https://hdl.handle.net/10016/30350)

**[Bachelor’s Thesis GitHub](https://github.com/j-moralejo-pinas/bt-rl-ad-robot/)**. My first attempt at teaching a car to drive using Deep Reinforcement Learning (DRL)—featuring Soft Actor-Critic (SAC), an autoencoder for feature extraction, and enough spaghetti code to make an Italian grandmother weep

**[CARLA_RL](https://github.com/j-moralejo-pinas/carla_rl)** This is a much more refined version, designed to help students quickly start their autonomous driving projects. It provides a CARLA environment with data collection, rewards, feature extraction, and a complete pipeline to train a baseline agent.
