# imdb-agent

## Setup & Installation

### 1. Create a Virtual Environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

> **Note:** The testing commands listed below assume you are running them from within this activated virtual environment.

## Testing Workflow

This project includes a test manager script to facilitate regression testing and output comparison.

### 1. Establish Baseline
Run this command to generate the baseline test reports. This represents the "known good" state of your code.
```bash
python scripts/test_manager.py --baseline
```
-   Generates `test_reports/baseline.xml` (Status)
-   Generates `test_reports/baseline.txt` (Full Output)

### 2. Run Tests & Compare
Run this command to execute current tests and compare them against the baseline.
```bash
python scripts/test_manager.py
```
-   Generates `test_reports/current.xml` (Status)
-   Generates `test_reports/current.txt` (Full Output)
-   Generates `test_reports/comparison_report.txt` (Comparison Table)
-   Prints the comparison report to the console.

### 3. Manual Output Comparison
To see exactly what changed in the test output (stdout), use `diff`:
```bash
diff test_reports/baseline.txt test_reports/current.txt
```

### 4. To run a specific test, use this format for your command:
```
python -m pytest tests/test_ambiguity.py::TestAmbiguityModule::test_high_ambiguity_trigger
```
---------------------------
## Core Architecture

### 1. Agent Design
-   **Type**: Zero-Shot React Description (`AgentType.ZERO_SHOT_REACT_DESCRIPTION`).
-   **Model**: `gpt-4o` (OpenAI).
-   **State Management**: 
    -   **Filtered Context** (`self.active_filter_titles`): Narrows down the search universe (e.g., "Only Comedy Movies").
    -   **Ambiguity State** (`self.pending_ambiguous_query`): Tracks pending queries that require user clarification.

### 2. Ambiguity Analysis
Before executing any tool, the agent evaluates the user's prompt for ambiguity using a dedicated LLM step.
-   **Scoring**: Assigns a score between 1-100.
    -   **1-20**: Clear (e.g., "Rating of The Godfather").
    -   **21-89**: Safe to guess (e.g., "Batman movies").
    -   **90-100**: Critical Ambiguity (e.g., "Best movie").
-   **Threshold**: If Score >= 90, the agent pauses execution and asks a clarification question.
-   **Resolution**: The user's response is combined with the original query to allow execution to proceed with full context.

### 3. Tools
The agent uses three primary tools to solve user queries:

#### A. `DatasetFilter` (Stateful)
-   **Purpose**: Creates a subset of the data to constrain subsequent searches.
-   **Function**: Generates and applies a pandas boolean mask (e.g., `df['Genre'].str.contains('Comedy')`).
-   **Output**: Updates `self.active_filter_titles` and returns the count of remaining movies.
-   **Use Case**: "Filter to Comedy movies", "Focus on movies from the 90s".

#### B. `PlotSearch` (Vector Search)
-   **Purpose**: Semantic search over movie plots using embeddings (FAISS).
-   **Logic**:
    1.  Checks if a Filtered Context exists (`active_filter_titles`).
    2.  If Active: Retrieves a large candidate set (`k=500`) and strictly filters them against the allowed titles.
    3.  If Inactive: Performs a standard similarity search (`k=50`).
-   **Output**: Top 5 most relevant movies with their plots.
-   **Use Case**: "Movies about death", "Aliens attacking earth".

#### C. `DataAnalyst` (Pandas Agent)
-   **Purpose**: Precise structured data operations.
-   **Function**: Executes Python/Pandas code on the `imdb_top_1000.csv` DataFrame.
-   **Use Case**: "What is the average rating?", "Count movies by Spielberg", "Sort by Year".

### 4. Suggestion Engine
After the main agent generates a response, a secondary process runs to enhance engagement:
-   **Extraction**: Uses an LLM to identify specific movie titles mentioned in the agent's text response.
-   **Logic**: Finds other movies in the dataset with similar metrics to the extracted titles (Meta_score ±10, IMDB_Rating ±1.0).
-   **Output**: Appends a "You might also like" list to the chat, identifying high-quality movies comparable to the ones discussed.
