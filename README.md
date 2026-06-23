# Zirex - LLM based modular assistant with agentic and workflow abilities

It's and AI chatbot using STT (Speech to Text) and TTS (Text to Speech) for audio conversation in real-time
using lightweight open-source LLM's and models for handling.

Basic architecture:

```
Listen for trigger -> STT -> LLM -> <background_workflows> -> TTS -> Play the answer as audio
```

---

### Stack

**XTTS2** - Model for converting text to speech (TTS) for making audio from LLM answers

**OpenAI Whisper** - Model for converting speech to text (STT) for making text from audio for asking
a question to LLM using the speech

**Llama 3.1 8B Q4** - LLM model, the brain of the workflow. On current stage, Llama just takes a query and makes an answer.
On the future, these functionalities will be added:
 1. Formatting standards: For example, order the LLM to use only json formatted answer, and use it for example
 to divide the code and the real answer text to avoid the TTS speak the code. It's awful so.
 2. Tools: Add tools to expand LLM's opportunities. First the **global search** will be added, and other tools
 depending on the development of the project.
 3. Advanced functionalities for agentic/workflow: Prompt chaining, prompt decomposition with parallel requests, new more smart
 model like **DeepSeek R1 B14 Q8** which should be used only for reasoning/hard questions, coding...


---

It's only project initialization but, there will be more and more features, so anyone could use it as local assistant
**FOR FREE** but, it's also needs a more powerful hardware like **RTX 3080 12GB** at least. If there will be 2 LLM's
like one for routine tasks and another like the same reasoning `DeepSeek`, it needs at least **24GB VRAM** so it's not
cheap. But, I'll try to keep on alive this project.
