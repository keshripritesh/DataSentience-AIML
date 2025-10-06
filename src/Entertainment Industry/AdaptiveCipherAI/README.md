## AdaptiveCipherAI: Neuro-Symbolic Codebreaking for Classical Ciphers

An advanced educational framework that procedurally generates classical ciphers and applies rule-based, statistical, and learning-based approaches to decrypt them without prior keys. Combines classical cryptanalysis (frequency analysis, IoC, chi-squared) with modern AI patterns (classification stubs, RL stubs) to explore how agents can learn cipher structure rather than brute forcing.

### Features
- Procedural dataset generation for Caesar, Monoalphabetic Substitution, and Vigenère
- Statistical feature extraction: unigrams/bigrams/trigrams, Index of Coincidence, entropy, chi-squared scoring
- Rule-based solvers: Caesar shift detection; substitution hill-climbing with Englishness scoring
- ML/RL extension points: cipher-type classifier stub and RL-style random search baseline
- Streamlit UI for quick encryption and solver demos

### Repository Structure
```
adaptivecipherai/
├─ ciphers/
├─ features/
├─ solvers/
├─ ui/
├─ tests/
└─ examples/
```

### Quickstart
```bash
python ciphers/generator.py --num_samples 200 --types caesar substitution vigenere --out examples/sample_ciphers.json

python -m pip install -r requirements.txt

pytest -q
```

### Streamlit Demo
```bash
streamlit run ui/streamlit_app.py
```

### CLI Examples
```bash
# ML stub
python solvers/ml_decoder.py --data examples/sample_ciphers.json --epochs 5

# RL stub (caesar)
python solvers/rl_agent.py --cipher "ZOLSS ZVD DVYK" --type caesar
```

### Roadmap
- Add Vigenère key length detection and Kasiski/Babbage methods
- Add proper n-gram language model scoring and beam search
- Introduce transformer-based seq2seq demo for substitution
- Implement actual RL policy with environment for mapping edits


