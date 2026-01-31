# Multi-Modal Misinformation Detection Pipeline

This module implements a modular pipeline architecture for multi-modal misinformation detection, organized by the 7 mandatory deliverables from PS 2.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MisinformationPipeline                          │
│                      (Main Orchestrator)                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Pipeline Stages                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │ D1: Input       │  │ D2: Cross-Modal │  │ D3: Context     │        │
│  │ Handler         │─▶│ Detector        │─▶│ Detector        │        │
│  │ (text+image)    │  │ (CLIP)          │  │ (web search)    │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │ D4: Synthetic   │  │ D5: Explanation │  │ D6: Robustness  │        │
│  │ Detector        │─▶│ Generator       │─▶│ Checker         │        │
│  │ (AI detection)  │  │ (Gemini)        │  │ (adversarial)   │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                         │
│  ┌─────────────────┐                                                   │
│  │ D7: Evaluation  │                                                   │
│  │ (metrics)       │                                                   │
│  └─────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Mandatory Deliverables Mapping

| Deliverable | Module | Description |
|-------------|--------|-------------|
| **D1** | `input_handler.py` | Multi-modal input handling (text + image and/or video) |
| **D2** | `cross_modal_detector.py` | Detection of cross-modal inconsistencies using CLIP |
| **D3** | `context_detector.py` | Identification of out-of-context media reuse |
| **D4** | `synthetic_detector.py` | Detection of AI-generated synthetic media |
| **D5** | `explanation_generator.py` | Natural language explanation generation |
| **D6** | `robustness.py` | Robustness against adversarial perturbations |
| **D7** | `evaluation.py` | Quantitative evaluation metrics |

## Usage

### Basic Usage (New Pipeline)

```python
from src.pipeline import MisinformationPipeline
from PIL import Image

# Create pipeline
pipeline = MisinformationPipeline()

# Run analysis
result = pipeline.run(
    text="Breaking: Massive earthquake hits Tokyo",
    image=Image.open("news_image.jpg")
)

# Access results
print(f"Verdict: {result.verdict}")
print(f"Truthfulness Score: {result.truthfulness_score}")
print(f"Explanation: {result.explanation}")
print(f"Cross-Modal Score: {result.cross_modal_score}")
print(f"AI Text Probability: {result.ai_text_probability}")
```

### Backward-Compatible Usage

```python
from src.analysis import detect_misinformation

result = detect_misinformation(
    text="Breaking news claim...",
    image=some_pil_image
)

# Same output format as before
print(result["verdict"])
print(result["truthfulness_score"])
```

### Custom Pipeline (Select Stages)

```python
from src.pipeline import MisinformationPipeline
from src.pipeline.stages import (
    InputHandlerStage,
    SyntheticDetectorStage,
    ExplanationGeneratorStage,
)

# Create pipeline with only specific stages
pipeline = MisinformationPipeline(stages=[
    InputHandlerStage(),
    SyntheticDetectorStage(),
    ExplanationGeneratorStage(),
])

result = pipeline.run(text="Some claim to verify")
```

## Module Structure

```
src/pipeline/
├── __init__.py              # Module exports
├── base.py                  # Base classes (PipelineStage, PipelineResult, etc.)
├── pipeline.py              # Main pipeline orchestrator
└── stages/
    ├── __init__.py          # Stage exports
    ├── input_handler.py     # D1: Multi-modal input handling
    ├── cross_modal_detector.py  # D2: Cross-modal inconsistency detection
    ├── context_detector.py  # D3: Out-of-context media detection
    ├── synthetic_detector.py    # D4: Synthetic media detection
    ├── explanation_generator.py # D5: Explanation generation
    ├── robustness.py        # D6: Robustness checks
    └── evaluation.py        # D7: Evaluation metrics
```

## Key Classes

### `PipelineInput`
Unified input structure for all stages:
- `text`: The claim/caption to analyze
- `image`: Optional PIL Image
- `video_path`: Optional path to video file
- `metadata`: Additional metadata dict

### `PipelineResult`
Complete output from pipeline execution:
- `verdict`: Final verdict string
- `truthfulness_score`: 0-100 score
- `explanation`: Human-readable explanation
- `evidence`: List of evidence points
- `stage_results`: Dict of individual stage results
- `cross_modal_score`, `ai_text_probability`, etc.

### `PipelineStage` (Abstract)
Base class for all pipeline stages:
- `stage_type`: The StageType enum value
- `name`: Human-readable name
- `description`: What this stage does
- `execute()`: Main execution method
- `should_skip()`: Conditional skip logic

## Extending the Pipeline

### Creating a Custom Stage

```python
from src.pipeline.base import PipelineStage, StageResult, StageType, PipelineInput

class CustomStage(PipelineStage):
    @property
    def stage_type(self) -> StageType:
        return StageType.EVALUATION  # or add new type
    
    @property
    def name(self) -> str:
        return "My Custom Stage"
    
    @property
    def description(self) -> str:
        return "Description of what this stage does"
    
    def execute(self, pipeline_input, previous_results) -> StageResult:
        # Your logic here
        return StageResult(
            stage_type=self.stage_type,
            success=True,
            data={"key": "value"}
        )
```

### Adding to Pipeline
```python
pipeline = MisinformationPipeline()
pipeline.stages.append(CustomStage())
```
