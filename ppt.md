```markdown
---
marp: true
theme: default
size: 16:9
---

# RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics

**Presenter**: [Your Name]
**Date**: [Date]

---

## Agenda
1. Motivation
2. Related Work on VLM
3. Methods
4. Results
5. Downstream Tasks
6. Core Codes

---

## Motivation
- **Precise Action Guidance**: Robots need to understand spatial affordances in images based on language instructions for precise actions.
- **Real - World Applications**: In scenarios like object manipulation, navigation, and scene understanding, robots require accurate keypoint prediction.
- **Limitations of Existing Models**: Current VLMs may not focus on spatial affordance prediction for robotics effectively.

![Overview](figures/overview.gif)

---

## Related Work on VLM
- **General VLMs**: Models like LLaVA have made significant progress in vision - language tasks, but they are not specifically tailored for spatial affordance prediction in robotics.
- **Robotics - Related Vision Models**: Some models focus on object detection or navigation, but lack the integration of language understanding for precise action guidance.

| Model | Focus | Limitations |
| --- | --- | --- |
| LLaVA | Vision - language understanding | Not specialized for spatial affordance in robotics |
| [Other Robotics Vision Models] | Object detection/Navigation | Lack language - driven spatial affordance prediction |

---

## Methods
### Overall Architecture
- **Language Model**: Based on existing large language models like Vicuna or Llama - 2.
- **Vision Tower**: Uses pre - trained vision encoders such as CLIP - L - 336px.
- **Projection Layer**: Different projection types (MLP2x, Linear) are used to align vision and language features.

### Multimodal Processing
- **Image Feature Extraction**: Extracts features from images using the vision tower.
- **Feature Fusion**: Fuses image features with language features through the projection layer.
- **Input Preparation**: Handles multimodal inputs, including image tokens and text, in the model's forward pass.

```python
# Example of forward pass in LlavaMptForCausalLM
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    images=None):

    input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
    
    return super().forward(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
```

---

## Results
### Evaluation on Where2Place Benchmark
- **Accuracy**: The model achieves [X]% accuracy in predicting spatial free - space references on challenging real - world images.
- **Visualization**: Visualizing the model's outputs shows that it can accurately pinpoint relevant points in the images.

```bash
# Command to generate results
python robopoint/eval/model_vqa.py \
    --model-path wentao-yuan/robopoint-v1-vicuna-v1.5-13b \
    --image-folder datasets/where2place/images \
    --question-file datasets/where2place/point_questions.jsonl \
    --answer-file output/robopoint-v1-vicuna-v1.5-13b.jsonl

# Command to compute accuracy
python robopoint/eval/summarize_vqa.py --answer output/robopoint-v1-vicuna-v1.5-13b.jsonl
```

### Comparison with Other Models
- **RoboPoint outperforms**: Compares favorably with other VLMs in terms of spatial affordance prediction accuracy.

---

## Downstream Tasks
- **Object Manipulation**: Robots can use the predicted keypoints to pick and place objects accurately.
- **Navigation**: Helps robots navigate through complex environments by understanding spatial affordances.
- **Scene Understanding**: Enables better understanding of scenes in terms of available free space and object locations.

---

## Core Codes
### Model Definition
```python
class LlavaMptForCausalLM(MptForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMptConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MptForCausalLM, self).__init__(config)

        self.transformer = LlavaMptModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
```

### Data Preparation
```python
def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources
```

---

## Conclusion
- **RoboPoint Contribution**: A novel VLM for spatial affordance prediction in robotics, providing precise action guidance.
- **Future Work**: Further improvement in accuracy, exploration of more complex downstream tasks, and optimization of the model architecture.

Thank You!

```

You can copy the above content and use Marp to generate the PPT. Note that you may need to adjust the content according to your actual presentation needs and the details of the "RoboPoint" paper.