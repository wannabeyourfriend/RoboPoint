# RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics
- Wentao Yuan, Jiafei Duan, Valts Blukis, et al.
- University of Washington, NVIDIA, Allen Institute for Artificial Intelligence, etc.
- arXiv:2406.10721v1 [cs.RO] 15 Jun 2024

## Motivation
### Spatial Reasoning in Daily Life and Robotics
- Spatial reasoning is fundamental to human intellectual processes and daily tasks, like grocery shopping.
- In robotics, waypoints based on pointing mimic human behavior, leading to intuitive plans.
- With the rise of VLMs, language instructions are used, but language lacks precision for guiding robot behavior. For example, "place the cup next to the plate" is hard for VLMs to execute accurately.

## Motivation
### Limitations of Existing VLMs in Robotics
- VLMs trained on large image - language datasets offer semantic understanding but struggle with precise spatial guidance.
- GPT-4o, a powerful VLM, has limited accuracy in real robot execution when dealing with spatial relations.

## Related Work
### Spatial Reasoning
- Many VQA benchmarks include spatial relation problems. Traditional methods like state estimation plus symbolic reasoning have poor generalization.
- SORNet shows zero-shot generalization in spatial reasoning with object prompts. SpatialVLM predicts spatial relations in metric space, but RoboPoint outperforms it in real - world tasks by locating affordances as points.

## Related Work
### Affordance Prediction
- Affordance defines how an object can be manipulated, going beyond visual properties.
- It can be represented in various ways, and RoboPoint uses 2D keypoint representation for easy conversion to language format.
- Many learning - based methods have shown the efficacy of affordance prediction in grasping and object placement.

## Related Work
### Zero-shot Language Models for Robotics
- Some works use language models as planners for robotic tasks with in - context learning, but they rely on pre - defined action primitives.
- VoxPoser generates 3D value maps, PIVOT iteratively samples and evaluates actions, and MOKA predicts action - specific keypoints. However, they all depend on external object detectors, unlike RoboPoint.

## Methods
### Spatial Affordance Prediction
- Problem formulation: Predict a set of target point coordinates in image space that satisfy language - indicated relations.
- Advantages: More precise than fuzzy language actions, and general enough for various robotic tasks like navigation, grasping, and placement.

## Methods
### Instruction Fine - tuning
- Build own dataset and fine - tune Vicuna - v1.5 - 13B model.
- Model components: Image encoder, MLP projector, language tokenizer, and transformer language model.
- Only update projector and transformer weights during fine - tuning.
- Achieves higher precision than baselines using in - context learning.

## Methods
### Co - finetuning with Synthetic Data
- Combine data from different sources: VQA data, LVIS data, object reference, and free space reference.
- Each data source contributes to the model's overall accuracy.
- Data quantity is crucial; performance drops when using only 10% of the data.

## Results
### Spatial Affordance Prediction
- Benchmarks: Evaluate on object reference (RoboRefIt) and free space reference (WHERE2PLACE) datasets.
- Baselines: Compare with Qwen - VL, LLaVA - NeXT, GPT - 4o, and SpaceLLaVa.
- Results: RoboPoint outperforms baselines significantly in accuracy. It can generalize to novel relation types, respect physical constraints, and maintain common sense knowledge.

## Results
### Quantitative Comparisons
| Dataset | Qwen - VL | LLaVA - NeXT - 34B | SpaceLLaVa | GPT - 4o | GPT - 4o - ICL | RoboPoint |
| --- | --- | --- | --- | --- | --- | --- |
| RoboRefIt | 24.08 ± 0.85 | 19.91 ± 0.92 | 21.30 ± 0.87 | 15.28 ± 1.27 | 9.01 ± 6.45 | 49.82 ± 0.52 |
| WHERE2PLACE | 10.49 ± 0.77 | 15.02 ± 0.88 | 11.84 ± 0.73 | 29.06 ± 1.33 | 14.46 ± 6.38 | 46.77 ± 0.45 |
| WHERE2PLACE (h) | 9.90 ± 0.22 | 14.76 ± 2.42 | 12.10 ± 1.36 | 27.14 ± 1.47 | 14.83 ± 4.68 | 44.48 ± 1.35 |

## Results
### Downstream Applications
- Real - World Manipulation: Set up 3 environments with 7 tasks. RoboPoint outperforms baselines like GPT - 4V in average success rate, enabling new capabilities like accurate object packing.
- Navigation: Tested in 3 room scenes with YouBot platform. RoboPoint outperforms PIVOT and GPT - 4V in 2 out of 3 scenarios.
- Augmented Reality: RoboPoint can provide natural language responses and visual action suggestions, helping users solve tasks like tic - tac - toe and getting to a carpool lane.

## Core Codes
### Instruction Tuning
```python
# RoboPoint is instruction - tuned from Vicuna - v1.5 - 13B
# with a ViT - L/14 336px image encoder pretrained with CLIP
# and a 2 - layer MLP projector
# Learning rate is set to 4e - 5, and it takes 40 hours on 16 A - 100 GPUs with batch size 16 per - GPU
```

## Core Codes
### Data Generation
```python
# Procedurally generate synthetic dataset
# Sample kitchen - like assets, place objects randomly on support surfaces
# Compute pairwise relations and sample points for object and free space reference
# Around 660K (image, relation) pairs are generated from 10K scenes
```

## Conclusion
### Summary of RoboPoint
- RoboPoint is a novel VLM for predicting spatial affordances based on relational language instructions.
- It overcomes limitations of current VLMs in robotics by integrating real - world and synthetic data.
- Achieves superior performance in complex tasks and has potential in augmented reality and robot navigation.

## Conclusion
### Limitations and Future Work
- RoboPoint lacks confidence estimates for point predictions and has uncontrollable output point numbers.
- Future work can focus on addressing these limitations and further improving the model's performance and generalization.