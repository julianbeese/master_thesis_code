# Master Thesis: Analyzing Rhetorical Framing in Brexit Debates in the House of Commons of the United Kingdom using Large Language Models: Temporal Trends and Structural Patterns 

This repository contains the complete codebase for a master thesis project analyzing how Brexit-related debates in the UK House of Commons are framed by different political parties, MPs, and constituencies. The project encompasses data collection, classification, annotation, machine learning model training, and comprehensive data analysis.

## Project Overview

This research project investigates framing patterns in UK parliamentary debates related to Brexit. It combines web scraping, natural language processing, machine learning, and statistical analysis to examine how different political actors frame Brexit-related discussions across temporal, geographical, and demographic dimensions.

## Repository Structure

The project is organized into four main components:

### 1. `data_scraping/` - Data Collection Pipeline

Web scraping and parsing infrastructure for collecting parliamentary debate data from official UK Parliament sources.

**Key Components:**
- **`scraper/`**: XML scraping from parliamentary websites
- **`parser/`**: XML parsing and data extraction
- **`classifier/`**: Initial Brexit-related content filtering
- **`database/`**: DuckDB-based data management system

**Main Scripts:**
- `scripts/run_full_scraping.py`: Complete scraping pipeline (scraping → parsing → database)
- `scripts/run_debates_scraper.py`: Scrape debate XML files
- `scripts/run_debates_parser.py`: Parse XML files into structured data

### 2. `data_classification/` - Content Classification & Annotation

Classification and annotation tools for identifying Brexit-related content and labeling frames.

**Key Components:**
- **`scripts/`**: Brexit classification using keyword matching and LLM APIs (Gemini, OpenAI)
- **`frame_classification/`**: Streamlit-based annotation interface for frame labeling
- **`data/processed/`**: Processed databases with classified debates and speeches

**Main Scripts:**
- `scripts/classify_brexit.py`: Hybrid keyword + LLM classification of debates
- `scripts/classify_brexit_with_gemini.py`: Gemini-based classification
- `scripts/classify_brexit_with_openai.py`: OpenAI-based classification
- `frame_classification/scripts/streamlit_annotation_railway.py`: Web-based annotation tool

**Features:**
- Two-stage classification: keyword filtering → LLM analysis
- Cost tracking for API usage
- Resume capability for interrupted classification runs
- Frame annotation interface with conflict resolution

### 3. `data_analysis/` - Statistical Analysis & Visualization

Comprehensive analysis notebooks examining framing patterns across multiple dimensions.

**Analysis Modules:**
- **`party/`**: Party-based frame analysis (Conservative, Labour, Liberal Democrat, SNP)
- **`demographical/`**: Demographic analysis of framing patterns
- **`geographical/`**: Constituency and regional frame distribution
- **`temporal/`**: Temporal distribution of frame observations
- **`eda/`**: Exploratory data analysis

**Key Scripts:**
- `scripts/aggregate_frames.py`: Aggregates frame labels from chunks to speeches and debates

### 4. `ml_training/` - Machine Learning Model Training

Training infrastructure for fine-tuning language models on frame classification tasks.

**Models:**
- **`bert/`**: BERT-based frame classification models
- **`mistral/`**: Mistral model fine-tuning with supervised fine-tuning (SFT)

**Features:**
- PEFT (Parameter-Efficient Fine-Tuning) support
- Weights & Biases integration for experiment tracking
- Docker support for cloud training (RunPod)
- GPU testing utilities

## Technology Stack

### Core Technologies
- **Python 3.10+**: Main programming language
- **DuckDB**: Analytical database for data storage and querying
- **Polars/Pandas**: Data manipulation and analysis
- **Streamlit**: Web-based annotation interface

### Machine Learning & NLP
- **Transformers (Hugging Face)**: Pre-trained language models
- **PEFT**: Parameter-efficient fine-tuning
- **Google Gemini API**: LLM-based classification
- **OpenAI API**: Alternative LLM classification

### Data Processing
- **BeautifulSoup4**: XML/HTML parsing
- **PyArrow**: Efficient data serialization
- **NumPy, Matplotlib, Seaborn**: Data analysis and visualization

### Infrastructure
- **Docker**: Containerization for ML training
- **Railway**: Deployment platform for annotation tool
- **PostgreSQL**: Database for Railway deployment

## Data Flow

```
1. Web Scraping (XML files)
   ↓
2. XML Parsing → Structured Data (DuckDB)
   ↓
3. Brexit Classification (Keywords + LLM)
   ↓
4. Frame Annotation (Streamlit interface)
   ↓
5. Data Aggregation (Chunks → Speeches → Debates)
   ↓
6. Statistical Analysis (Party, Demographics, Geography, Temporal)
   ↓
7. ML Model Training (BERT, Mistral)
```

## Database Schema

The main database (`thesis_final.duckdb`) contains:

- **`debates`**: Debate metadata (date, topic, URL)
- **`topics`**: Sub-topics within debates
- **`speeches`**: Individual speech records with text and metadata
- **`chunks`**: Text chunks with frame labels
- **`MPS`**: MP metadata (party, constituency, demographics, Brexit vote)
- **`constituencies`**: Constituency information

## Key Features

### Cost-Efficient Classification
- Two-stage filtering reduces LLM API costs
- Configurable cost limits
- Resume capability for interrupted runs

### Scalable Annotation
- Web-based Streamlit interface
- Conflict resolution for multi-annotator scenarios
- Railway deployment for remote access

### Comprehensive Analysis
- Multi-dimensional analysis (party, demographics, geography, time)
- Frame aggregation across hierarchy levels
- Statistical testing and visualization

## Project Status

This repository represents a complete research pipeline from data collection to analysis. The codebase has been used to:

- Scrape and parse thousands of parliamentary debates
- Classify Brexit-related content using hybrid keyword+LLM approach
- Annotate text chunks with frame labels
- Train and evaluate ML models for frame classification
- Conduct comprehensive statistical analysis of framing patterns

## Contributing

This is a master thesis project repository. For questions or issues, please contact the repository owner.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: This project processes large amounts of parliamentary data. Ensure you have sufficient disk space and comply with data usage terms when scraping parliamentary websites.

