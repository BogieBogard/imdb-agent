import os
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS

class MovieAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        # os.environ["OPENAI_API_KEY"] = api_key # Not needed for Ollama
        
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
        # For local models, we can't set the max limit to None. Local models can't handle large dataframes.
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_colwidth', None)
        
        # Initialize LLM
        self.llm = ChatOllama(temperature=0, model="gemma3:12b")
        
        # Initialize Vector Store
        self.vector_store = self._load_or_create_vector_store()
        
        self.active_filter_titles = None  # State for "Filter-First"
        self.pending_ambiguous_query = None # State for ambiguity clarification

        # Create Tools
        self.pandas_agent_executor = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,
            # agent_type="tool-calling" failed with NotImplemnetedError for ChatOllama in this version.
            # Switching to generic zero-shot ReAct.
            agent_type="zero-shot-react-description"
        )
        
        self.tools = [
            Tool(
                name="DatasetFilter",
                func=self._run_dataset_filter,
                description="Useful for narrowing down the search scope *before* performing a plot search (e.g., 'Filter to Comedy movies only'). This tool filters the dataset and saves the filtered context for subsequent PlotSearch calls. Input should be the filtering criteria."
            ),
            Tool(
                name="DataAnalyst",
                func=self._run_pandas_agent,
                description="Useful for questions involving sorting, counting, math, averages, or structured data queries (e.g., 'Top 5 movies by rating', 'Movies released after 2010'). Input should be the full natural language query."
            ),
            Tool(
                name="PlotSearch",
                func=self._vector_search,
                description="Useful for finding movies based on plot descriptions, themes, or keywords (e.g., 'Movies about death', 'Alien invasion plots'). NOTE: This tool uses the filtered context set by DatasetFilter if available."
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
            return self.pandas_agent_executor.invoke(f"{query} Answer in a full sentence.")
        except Exception as e:
            return f"Error running pandas analysis: {str(e)}"
            
    def _run_dataset_filter(self, query):
        """
        Generates a pandas boolean mask from the query, applies it, 
        and saves the resulting titles to self.active_filter_titles.
        """
        try:
            # Prompt to get the boolean mask expression
            prompt = f"""
            You are a pandas expert. 
            DataFrame `df` has columns: {list(self.df.columns)}.
            
            Write a Python pandas filtering expression/mask to select/filter data based on: "{query}".
            Return ONLY the expression. Do not assign it to a variable.
            
            CRITICAL RULES:
            1. Use bitwise operators `&` (AND) and `|` (OR) for multiple conditions.
            2. WRAP EACH CONDITION IN PARENTHESES. Example: `(df['A'] > 1) & (df['B'] == 2)`
            3. Use `.str.contains()` for text search with `case=False, na=False`.
            
            Examples:
            Query: "Comedy movies" -> df['Genre'].str.contains('Comedy', case=False, na=False)
            Query: "Movies after 2000" -> df['Released_Year'] > 2000
            Query: "Rating > 8" -> df['IMDB_Rating'] > 8.0
            Query: "Steven Spielberg sci-fi movies" -> (df['Director'] == 'Steven Spielberg') & (df['Genre'].str.contains('Sci-Fi', case=False, na=False))
            Query: "Action movies with rating > 7.5" -> (df['Genre'].str.contains('Action', case=False, na=False)) & (df['IMDB_Rating'] > 7.5)
            """
            
            from langchain.schema import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            expression = response.content.strip().strip('`').replace('python', '').strip()
            
            # Safety / Sandbox check could go here, but allow_dangerous_code is set on the other agent anyway.
            # We assume the LLM produces a valid pandas mask.
            
            # Validating that the expression starts with df
            if not expression.startswith("df") and not expression.startswith("(df"):
                return "Failed to generate a valid filtering expression."
                
            # Execute
            # strict context with only 'df'
            local_scope = {'df': self.df}
            mask = eval(expression, {"__builtins__": None}, local_scope)
            
            filtered_df = self.df[mask]
            
            if filtered_df.empty:
                return "The filter resulted in 0 movies. Please relax the criteria."
                
            self.active_filter_titles = set(filtered_df['Series_Title'].tolist())
            
            return f"Filter applied successfully. {len(self.active_filter_titles)} movies selected. Context set for PlotSearch."
            
        except Exception as e:
            return f"Error applying filter: {str(e)}"

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
            
            # 2. Determine Scope & Candidates
            # If we have an active filter, we broaden the search to find hits WITHIN the filter
            if self.active_filter_titles:
                 k_candidates = 150  # Larger net
            else:
                 k_candidates = 50   # Standard net
            
            docs = self.vector_store.similarity_search(query, k=k_candidates)
            
            # 3. Filter results
            filtered_docs = []
            for doc in docs:
                movie_year = doc.metadata.get('year')
                movie_title = doc.metadata.get('title')
                
                if not movie_year: 
                    continue
                    
                keep = True
                
                # Apply Active Filter
                if self.active_filter_titles:
                    if movie_title not in self.active_filter_titles:
                        keep = False
                
                # Apply Extracted Constraints (Year)
                if keep:
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
                return "No movies found matching the plot and criteria."
                
            return "\n\n".join([f"Title: {d.metadata['title']} ({d.metadata['year']})\nPlot: {d.page_content}" for d in final_docs])
            
        except Exception as e:
            return f"Error searching plots: {str(e)}"

    def _load_or_create_vector_store(self):
        embeddings = OllamaEmbeddings(model="qwen3-embedding:latest")
        index_path = "faiss_index_qwen_latest"
        
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

    def _check_ambiguity(self, query):
        """
        Evaluates the ambiguity of a query using the LLM.
        Returns a dictionary: {"score": int, "clarification": str}
        """
        print(f"[DEBUG] Starting _check_ambiguity for query: '{query}'")
        try:
            prompt = f"""
            You are an ambiguity detector for a movie database agent.
            Analyze the following user query for ambiguity.
            
            Query: "{query}"
            
            Determine an "ambiguity_score" between 1 and 100.
            - 1-20: Very clear (e.g., "What is the rating of The Godfather?", "Who directed Inception?")
            - 21-89: Some ambiguity but safe to guess (e.g., "Batman movies" -> likely implies any movie with Batman). DEFAULT TO THIS RANGE if you can make a reasonable guess.
            - 90-100: CRITICAL ambiguity, IMPOSSIBLE to proceed without clarification (e.g., user asks for "the best movie" with absolutely no criteria).
            
            If score >= 90, provide a "clarification_question" to ask the user.
            
            Return in JSON format:
            {{
                "score": <int>,
                "clarification": "<string>"
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that detects ambiguity in queries."},
                {"role": "user", "content": prompt}
            ]
            
            print("[DEBUG] Invoking LLM for ambiguity...")
            response = self.llm.invoke(messages)
            print("[DEBUG] LLM response received.")
            content = response.content.strip()
            
            import json
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            result = json.loads(content)
            print(f"[Ambiguity Analysis] Query: '{query}' | Score: {result.get('score')} | Clarification: '{result.get('clarification')}'")
            return result
        except Exception as e:
            print(f"[ERROR] Ambiguity check failed: {e}")
            # Fallback if detection fails: assume no ambiguity
            return {"score": 0, "clarification": ""}

    def run(self, query):
        print(f"[DEBUG] agent.run called with: '{query}'")
        # 1. Check if we are waiting for a clarification response
        if self.pending_ambiguous_query:
            print("[DEBUG] Handling pending ambiguous query response.")
            # The current 'query' is the user's answer to our clarification question
            combined_query = f"User asked: '{self.pending_ambiguous_query}'. When asked for clarification, they replied: '{query}'. Answer the original question with this new context."
            # Reset state
            self.pending_ambiguous_query = None
            # Proceed with the combined query
            return self.master_agent.run(combined_query)
            
        # 2. Check for ambiguity on new queries
        ambiguity_result = self._check_ambiguity(query)
        
        if ambiguity_result.get("score", 0) >= 90:
            self.pending_ambiguous_query = query
            clarification = ambiguity_result.get("clarification", "Could you please clarify your request?")
            # Remove potential code block formatting and escape dollar signs (Streamlit treats $ as math)
            return clarification.replace("`", "").replace("$", "\\$")
            
        # 3. Normal execution
        return self.master_agent.run(query)

    def _extract_movies_from_text(self, text):
        """
        Uses LLM to identify movie titles mentioned in a text response.
        """
        try:
            prompt = f"""
            Identify the exact movie titles mentioned in the following text.
            Text: "{text}"
            
            Return ONLY a JSON list of strings. Example: ["The Godfather", "Inception"]
            If no movies are mentioned, return [].
            """
            from langchain.schema import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            content = response.content.strip()
            import json
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
                
            titles = json.loads(content)
            return titles if isinstance(titles, list) else []
        except Exception as e:
            print(f"[ERROR] Failed to extract movies: {e}")
            return []

    def _find_similar_movies(self, titles):
        """
        Finds movies with similar Meta_score and IMDB_Rating.
        """
        if not titles:
            return []
            
        suggestions = []
        seen_titles = set(titles) # Don't recommend the same movies
        
        # Get stats for the mentioned movies
        target_movies = self.df[self.df['Series_Title'].isin(titles)]
        
        if target_movies.empty:
            return []

        avg_meta = target_movies['Meta_score'].mean()
        avg_imdb = target_movies['IMDB_Rating'].mean()
        
        # Define similarity range
        meta_margin = 10
        imdb_margin = 1.0
        
        # Filter for similar movies
        candidate_mask = (
            (self.df['Meta_score'].between(avg_meta - meta_margin, avg_meta + meta_margin)) &
            (self.df['IMDB_Rating'].between(avg_imdb - imdb_margin, avg_imdb + imdb_margin))
        )
        
        candidates = self.df[candidate_mask].sort_values('IMDB_Rating', ascending=False)
        
        for _, row in candidates.iterrows():
            if row['Series_Title'] not in seen_titles:
                suggestions.append(f"**{row['Series_Title']}** (IMDB: {row['IMDB_Rating']}, Meta: {row['Meta_score']})")
                seen_titles.add(row['Series_Title'])
                
            if len(suggestions) >= 3:
                break
                
        return suggestions

    def get_suggestions(self, response_text):
        """
        Public method to generate suggestions based on the last response.
        """
        try:
            titles = self._extract_movies_from_text(response_text)
            if not titles:
                return None
                
            similar_movies = self._find_similar_movies(titles)
            
            if not similar_movies:
                return None
                
            return "### 🎥 You might also like:\n" + "\n".join([f"- {s}" for s in similar_movies])
        except Exception as e:
            print(f"[ERROR] Suggestion generation failed: {e}")
            return None

