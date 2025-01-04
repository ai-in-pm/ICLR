# ICLR: In-Context Learning of Representations

This repository contains an implementation inspired by the ICLR paper on dynamic representation learning and semantic reorganization in graph-based tasks. The implementation features advanced analysis tools, visualization capabilities, and LLM-powered semantic understanding.

The development of this GitHub Repository was inspired by the "ICLR: IN-CONTEXT LEARNING OF REPRESENTATIONS" Paper. To read the entire paper, visit https://arxiv.org/pdf/2501.00070 

## Features

### Core Capabilities
- Dynamic representation learning on graph structures
- Phase transition analysis and critical behavior detection
- Semantic prior analysis and influence quantification
- Advanced visualization of representation landscapes

### LLM Integration
- OpenAI GPT-4 powered semantic analysis
- Embedding generation using text-embedding-ada-002
- Intelligent representation interpretation
- Semantic structure insights

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ICLR.git
cd ICLR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY="your-api-key-here"
```

## Usage

Run the demo script to see the agent in action:
```bash
python examples/demo.py
```

The demo will:
1. Execute graph tracing tasks on different structures
2. Analyze phase transitions with LLM insights
3. Generate semantic prior analysis
4. Create visualizations in the `examples/outputs` directory

![image](https://github.com/user-attachments/assets/d8decd84-8c0a-460c-9f7d-9024c2b8462f)
![image](https://github.com/user-attachments/assets/c4108b95-c29b-4d65-9052-1d5954b07680)
![image](https://github.com/user-attachments/assets/92c3bcf3-ed94-44aa-a7ea-0b8d33b726df)
![image](https://github.com/user-attachments/assets/9f5c5c74-412b-4d1a-a4ef-710191f4fe3f)
![image](https://github.com/user-attachments/assets/e3b5658b-018e-46ef-bd4a-d67dc66fb2f1)




## Project Structure

```
ICLR/
├── src/
│   ├── models/
│   │   ├── iclr_agent.py      # Core agent implementation
│   │   └── llm_interface.py   # OpenAI API integration
│   ├── analysis/
│   │   ├── phase_transition.py # Phase transition analysis
│   │   └── semantic_priors.py  # Semantic prior analysis
│   └── visualization/
│       └── advanced_visualizer.py # Visualization tools
├── examples/
│   ├── demo.py                # Demo script
│   └── outputs/               # Generated visualizations
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Features in Detail

### Phase Transition Analysis
- Computation of order parameters using mutual information
- Detection of critical points in representation organization
- Analysis of scaling behavior pre and post-transition
- LLM-powered interpretation of phase transitions

### Semantic Prior Analysis
- Analysis of structured semantic priors (weekdays, months, numbers)
- Correlation computation between learned representations and priors
- GPT-4 powered insights into semantic structure
- Visualization of semantic influence patterns

### Advanced Visualization
- 3D representation landscapes using PCA/t-SNE
- Energy landscape visualization
- Representation flow visualization
- Interactive visualization options

## Dependencies
- NumPy (≥1.21.0)
- Matplotlib (≥3.4.3)
- NetworkX (≥2.6.3)
- Scikit-learn (≥0.24.2)
- Pandas (≥1.3.0)
- Seaborn (≥0.11.2)
- OpenAI (≥1.0.0)
- Python-dotenv (≥0.19.0)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this implementation in your research, please cite the original ICLR paper:
[Link to paper](https://arxiv.org/abs/paper_id)

## Acknowledgments
- Thanks to the authors of the ICLR paper for their groundbreaking research
- OpenAI for providing the GPT-4 and embedding models
- The open-source community for the excellent scientific computing tools
