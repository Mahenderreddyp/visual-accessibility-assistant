# 🦮 Visual Accessibility Assistant

> **AI-powered visual assistant for visually impaired users, running 100% on-device for privacy and offline functionality.**

![Demo](https://via.placeholder.com/800x400?text=Project+Demo+Snapshot)

## 🎯 Project Overview

* **What**: A mobile app that uses AI to answer questions about images in real-time.
* **Who**: Designed for blind and visually impaired users.
* **How**: Fine-tuned **Qwen2.5-VL-3B** model deployed via Ollama on edge devices.
* **Why**: Complete privacy (no cloud processing), works offline, <2s latency.

### Example Use Cases
| Image Type | User Question | AI Answer |
| :--- | :--- | :--- |
| **Pantry Item** | "What is this?" | *"Campbell's Chicken Noodle Soup, 10.75 oz"* |
| **Medicine** | "Is this safe to take?" | *"Ibuprofen 200mg, expires May 2026"* |
| **Street Sign** | "What does this sign say?" | *"Stop sign - octagonal red sign"* |

---

## 🏗️ Architecture

```mermaid
graph LR
    A[VizWiz Dataset<br/>31K images] -->|Fine-Tune| B(Unsloth Training<br/>Qwen2.5-VL-3B)
    B -->|Quantize| C{GGUF Conversion<br/>4-bit / INT4}
    C -->|Deploy| D[Edge Device<br/>Ollama Runtime]
    D -->|Inference| E[Mobile App<br/>Camera + Voice]