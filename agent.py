import os
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS

class MovieAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Load Data
        # Using a valid path assuming execution from repo root
        self.df = pd.read_csv("imdb_top_1000.csv")
        
        # Normalize columns for easier pandas usage
        self.df.columns = [c.replace(' ', '_') for c in self.df.columns]

        # --- Data Cleaning ---
        # Convert Released_Year to numeric
        self.df['Released_Year'] = pd.to_numeric(self.df['Released_Year'], errors='coerce')

        # Clean Gross column: remove commas and convert to float
        if self.df['Gross'].dtype == 'object':
             self.df['Gross'] = self.df['Gross'].str.replace(',', '').astype(float)
        self.df['Gross'] = self.df['Gross'].fillna(0)

        # Ensure numeric types for other key columns
        self.df['No_of_Votes'] = pd.to_numeric(self.df['No_of_Votes'], errors='coerce').fillna(0)
        self.df['Meta_score'] = pd.to_numeric(self.df['Meta_score'], errors='coerce').fillna(0)
        
        # --- Pandas Display Options ---
        # Prevent truncation that causes hallucinations
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        # Initialize Vector Store
        self.vector_store = self._load_or_create_vector_store()
        
        # Create Tools
        self.pandas_agent_executor = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type="openai-tools"
        )
        
        self.tools = [
            Tool(
                name="DataAnalyst",
                func=self._run_pandas_agent,
                description="Useful for questions involving sorting, filtering, counting, math, averages, or structured data queries (e.g., 'Top 5 movies by rating', 'Movies released after 2010'). Input should be the full natural language query."
            ),
            Tool(
                name="PlotSearch",
                func=self._vector_search,
                description="Useful for finding movies based on plot descriptions, themes, or keywords (e.g., 'Movies about death', 'Alien invasion plots'). NOTE: This tool DOES NOT filter by release year, rating, or other structured data. For queries involving dates/ratings AND plot/theme, use DataAnalyst to filter first or filter the results of this search using DataAnalyst."
            )
        ]
        
        self.master_agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def _run_pandas_agent(self, query):
        try:
            return self.pandas_agent_executor.invoke(query)
        except Exception as e:
            return f"Error running pandas analysis: {str(e)}"

    def _extract_year_constraints(self, query):
        """
        Uses an LLM to extract year filtering constraints from the query.
        Returns a dictionary: {'type': 'before'|'after'|'exact'|'range'|'none', 'year': int, 'year_end': int}
        """
        try:
            # Simple prompt to extract year
            messages = [
                {"role": "system", "content": "You are a helper that extracts year constraints from movie search queries."},
                {"role": "user", "content": f"""
                Analyze the following query and extract the year constraint.
                Query: "{query}"
                
                Respond in JSON format with these keys: 
                - "type": "before", "after", "exact", "range", or "none"
                - "year": integer year (for before/after/exact/range start)
                - "year_end": integer year (only for range end, else null)
                
                Examples:
                "Movies before 1990" -> {{"type": "before", "year": 1990, "year_end": null}}
                "80s movies" -> {{"type": "range", "year": 1980, "year_end": 1989}}
                "Movies from 2005" -> {{"type": "exact", "year": 2005, "year_end": null}}
                "Scary movies" -> {{"type": "none", "year": null, "year_end": null}}
                """}
            ]
            
            # Using the existing LLM (gpt-4o) for extraction
            response = self.llm.invoke(messages)
            import json
            # Extract JSON from response (naive parsing, stripping markdown code blocks if any)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            return json.loads(content)
        except Exception:
            return {"type": "none"}

    def _vector_search(self, query):
        try:
            # 1. Extract constraints
            constraint = self._extract_year_constraints(query)
            
            # 2. Fetch candidates (fetch many to allow for filtering)
            k_candidates = 50
            docs = self.vector_store.similarity_search(query, k=k_candidates)
            
            # 3. Filter results
            filtered_docs = []
            for doc in docs:
                movie_year = doc.metadata.get('year')
                if not movie_year: # Include if year key missing (safety) or keep? Let's drop if invalid.
                    continue
                    
                keep = True
                c_type = constraint.get('type')
                c_year = constraint.get('year')
                c_end = constraint.get('year_end')
                
                if c_type == 'before' and c_year:
                    if not (movie_year < c_year): keep = False
                elif c_type == 'after' and c_year:
                    if not (movie_year > c_year): keep = False
                elif c_type == 'exact' and c_year:
                    if not (movie_year == c_year): keep = False
                elif c_type == 'range' and c_year and c_end:
                    if not (c_year <= movie_year <= c_end): keep = False
                
                if keep:
                    filtered_docs.append(doc)
            
            # 4. Return top 5
            final_docs = filtered_docs[:5]
            
            if not final_docs:
                return "No movies found matching the plot and year criteria."
                
            return "\n\n".join([f"Title: {d.metadata['title']} ({d.metadata['year']})\nPlot: {d.page_content}" for d in final_docs])
            
        except Exception as e:
            return f"Error searching plots: {str(e)}"

    def _load_or_create_vector_store(self):
        embeddings = OpenAIEmbeddings()
        index_path = "faiss_index"
        
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating Vector Store... this may take a moment.")
            texts = self.df['Overview'].fillna("").tolist()
            titles = self.df['Series_Title'].fillna("Unknown").tolist()
            years = self.df['Released_Year'].fillna(0).astype(int).tolist()
            
            metadatas = [{"title": t, "year": y} for t, y in zip(titles, years)]
            
            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            vectorstore.save_local(index_path)
            return vectorstore

    def run(self, query):
        return self.master_agent.run(query)
