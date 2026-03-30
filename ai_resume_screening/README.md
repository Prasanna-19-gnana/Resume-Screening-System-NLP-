# Advanced AI Resume Screening Pipeline (v2.0)

A modular, production-grade PDF intelligent parsing and semantic scoring engine.

## 🚀 The Upgrade

This architecture replaces a faulty monolithic architecture with a **true scalable NLP micro-service workflow**:
1. **Raw Document Intake via PyMuPDF**: It extracts text safely without garbage blocks.
2. **Precision Regex Section Parsing**: Divides the resume specifically into headers like `projects`, `skills`, and `experience`.
3. **Smart Embedded Reconstruction**: Actively removes `education` and `fluff` from the semantic context window, limiting the string to a max dense 800 tokens. This drastically improves Sentences-Transformers embedding relevance.
4. **Multi-Label Probability Tiers**: The SVM classifier now returns full mathematical alignment probabilities (`"RoleA": 0.82`) instead of just a flat "RoleA"!
5. **Dynamic Scoring Weights Rule**: The logic now actively rewards technologically savvy candidates by dropping semantic penalty weights dynamically `if skill_score > 0.9`.

## ⚙️ How to Test It

Ensure the `ai_resume_screening` environment dependencies are fully aligned.
`(pip install -r requirements.txt PyMuPDF python-docx reportlab)`

### 1. Launch the Pipeline Brains
```bash
cd ai_resume_screening
../venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Stream Real Data (2-Step API Workflow)
In a separate terminal, execute our end-to-end integration test. This automatically creates a mock PDF loaded with Python, MERN, and NLP traits, posts it to the `/upload-resume` Endpoint, retrieves the JSON, then evaluates it against an NLP Engineer Job Description via `/match`:

```bash
cd ai_resume_screening
../venv/bin/python test_pipeline.py
```

### Expected Output Output:
```text
🎯 Final Multi-Layer Score:  76.5 / 100
🔮 Predicted Role Name:      Developer
🧠 Semantic Similarity:      0.63
🛠️ Skills Match Score:       0.95
   - Matched: python, machine learning, nlp, rest api, mern
```
