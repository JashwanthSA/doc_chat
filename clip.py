from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List, Dict
from PIL import Image
import io
from dotenv import load_dotenv
import os
import litellm
import base64

load_dotenv()

# CLIP model name (Hugging Face)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Threshold (tunable) for deciding "graph/workflow" vs not
CLIP_SIM_THRESHOLD = 0.20  # tune this on examples (0.0..1.0)

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "growth chart", "bar chart", "diagram", "flow chart",
    "graph", "just image", "pie chart", "table"
]
GRAPH_CLASSES = {"growth chart", "bar chart", "diagram", "flow chart", "graph", "pie chart", "table"}


def _clip_features_to_tensor(features) -> torch.Tensor:
    """CLIP get_*_features may return a tensor (older transformers) or BaseModelOutputWithPooling (newer)."""
    if isinstance(features, torch.Tensor):
        return features
    pooler = getattr(features, "pooler_output", None)
    if pooler is not None:
        return pooler
    raise TypeError(f"Unexpected CLIP feature type: {type(features)}")


def init_clip_model(model_name: str = CLIP_MODEL_NAME):
    """Load CLIP model + processor (cpu or cuda)."""
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device

def init_vlm_model() -> str:
    """Return the VLM model string for LiteLLM from environment. Must be a vision-capable model."""
    vlm_model = os.getenv("LITELLM_VLM_MODEL", "gpt-4o")
    return vlm_model

def score_image_against_labels(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    image_pil: Image.Image,
    labels: List[str],
) -> Dict[str, float]:
    """
    Return a dictionary label -> similarity score between image and label text.
    Uses CLIP image + text encoders and cosine similarity.
    """
    # Preprocess
    inputs = processor(text=labels, images=image_pil, return_tensors="pt", padding=True).to(device)
    # Get image and text features
    with torch.no_grad():
        image_embeds = _clip_features_to_tensor(
            model.get_image_features(**{k: inputs[k] for k in ["pixel_values"]})
        )
        text_embeds = _clip_features_to_tensor(
            model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
        )
    # Normalize
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    # Compute cosine similarities
    # image_embeds is (1, dim) when single image, text_embeds is (N, dim)
    sims = (text_embeds @ image_embeds.T).squeeze(-1)  # (N,)
    scores = sims.cpu().numpy().tolist()
    return {label: float(score) for label, score in zip(labels, scores)}


def classify_image_graph_or_not(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    image_bytes: bytes,
    threshold: float = CLIP_SIM_THRESHOLD,
) -> bool:
    """
    Returns True if image is likely a workflow diagram or graph.
    Uses zero-shot labels tuned to typical words describing diagrams/graphs.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        # If failing to open, consider as not a graph
        return False

    graph_labels = ['chart', 'bar chart', 'line chart', 'pie chart', 'histogram', 'scatter plot', 'workflow diagram', 'flowchart', 'network diagram', 'process diagram', 'system architecture diagram', 'organizational chart', 'data table', 'spreadsheet', 'excel sheet', 'grid of data', 'structured form', 'report table', 'data matrix', 'table with rows and columns']
    other_labels = [
        "photograph", "person photo", "group photo", "selfie", "landscape photo",
        "icon", "decorative image", "logo", "background pattern", "illustration",
        "company logo", "website screenshot", "banner image", "button icon", "app interface",
        "product photo", "vehicle", "building", "room interior", "street scene",
        "clipart", "symbol", "badge", "mascot", "painting"
    ]

    # Score image vs both label sets
    positive_scores = score_image_against_labels(model, processor, device, image, graph_labels)
    negative_scores = score_image_against_labels(model, processor, device, image, other_labels)

    # Heuristic: if max positive score substantially > max negative score and above threshold -> graph/workflow
    max_pos = max(positive_scores.values())
    max_neg = max(negative_scores.values())
    print(f"Max positive score: {max_pos}, Max negative score: {max_neg}")
    print(f"Threshold: {threshold}")
    is_graph = (max_pos >= threshold) and (max_pos > max_neg - 0.05)
    return is_graph

def describe_image(image_bytes: bytes, vlm_model: str):
    """
    Generate a descriptive summary of an image (graph/diagram/workflow) using LiteLLM.
    
    Args:
        image_bytes: Raw image bytes
        vlm_model: LiteLLM model string for the vision-capable model
    
    Returns:
        str: Natural language description of the image
    """
    system_prompt = """
    You are a Data Analyst and Technical Writer. Analyze a visual element (graph, chart, workflow, or diagram) and produce a factual, compact summary within 600 tokens.

### TASK
1. Identify the type of visual (e.g., bar chart, workflow, timeline).
2. Describe its purpose and main message.
3. Summarize key relationships, patterns, or processes shown.
4. Mention crucial quantitative or categorical details if visible.
5. Keep descriptions clear, structured, and under 2500 characters.

### OUTPUT FORMAT
**Summary:** A single, coherent paragraph describing:
- The visual’s type and subject.
- The main trend, flow, or relationship.
- Key components or comparisons in business or analytical terms.

### DO NOT
- Do not speculate or infer hidden meanings.
- Do not use lists or conversational phrases.
- Do not exceed 600 tokens or 2500 characters.
    """
    
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    try:
        response = litellm.completion(
            model=vlm_model,
            messages=[
                {"role": "system", "content": system_prompt}, 
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"VLM request failed: {e}")

def describe_slides(image_bytes: bytes, vlm_model: str):
    """
    Generate a detailed description of a presentation slide image using LiteLLM.

    Args:
        image_bytes: Raw slide image bytes
        vlm_model: LiteLLM model string for the vision-capable model

    Returns:
        str: Thorough natural-language and markdown capture of the slide content
    """
    system_prompt = """
You are an expert Technical Writer and Data Analyst. Describe this presentation slide for use in a **chatbot / RAG context**: capture the **substantive content** (what the slide communicates) accurately and completely, without filler.

### TASK
1. Cover all **meaningful** on-slide text, numbers, chart data, workflow logic, and table cell values—nothing important should be dropped.
2. Weave **titles, body copy, charts, and workflows** together in **plain prose** (see OUTPUT FORMAT). Quote short phrases verbatim when needed for precision; otherwise paraphrase clearly.
3. For **charts**: in prose, state type, axes (names/units if visible), series/categories, legend, visible values or labels, trends, and comparisons. If a number is estimated from the image, say it is approximate.
4. For **workflows / diagrams**: in prose, explain the process—steps or nodes, readable labels, flow direction, branches, and relationships. Include every readable label that matters to the message.
5. For **photos, icons, logos, screenshots**: only mention what they add to the message (subject, product, readable UI text)—not decorative detail.

### OUTPUT FORMAT
**Heading:** On the very first line, output a single `##` Markdown heading that concisely names this slide's topic (e.g., `## AWS Migration Overview`). Do not use generic names like "Slide 1" or "Summary". This heading must stand alone on its own line.

**Summary:** Immediately after the heading, write **two to four cohesive paragraphs** of continuous prose (no `###` section headings, no labels like "On-slide text" or "Layout"). The first paragraph should orient the reader to the slide's main topic; following paragraphs should carry remaining text, chart, and workflow detail in natural reading order.

**Tables:** If and only if the slide contains a table (or tables), **after** the prose **Summary** (or between paragraphs if one table clearly belongs between two ideas), output each table as its own **GitHub-flavored Markdown pipe table** that **one-to-one** matches visible rows, columns, headers, and cell text. Do not wrap tables in code fences unless cells contain pipes that break parsing. Do not add a second prose "wrapper" around tables beyond a single short bridging sentence if truly needed.

If there are **no** tables, output the **Heading** and **Summary** paragraphs—nothing else.

### DO NOT
- Do **not** describe PowerPoint **layout, design, or metadata**: slide masters, placeholders, alignment, whitespace, margins, fonts, font sizes, colors, backgrounds, "title on top left", grid position, aspect ratio, or similar—unless a visual choice is itself the **message** (e.g., a red alert callout whose meaning is "warning").
- Do **not** use topic subheadings, bullet lists for the main narrative, or meta-commentary ("this slide shows", "in the image").
- Do **not** invent values or text; if something is illegible, say so briefly in prose.

### CONSTRAINTS
- Prefer **dense, readable prose** over length; stay within the model’s reply budget while still preserving critical numbers and labels.
    """
    
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    try:
        response = litellm.completion(
            model=vlm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ],
                },
            ],
            max_tokens=2500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"VLM request failed: {e}")