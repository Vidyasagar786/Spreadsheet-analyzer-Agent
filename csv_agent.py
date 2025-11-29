import streamlit as st
import pandas as pd
import os
import re
import ast
import logging
import io
import asyncio
from typing import Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
clog = logging.getLogger(__name__)

load_dotenv()

class CodeGenerationError(Exception): pass
class CodeExecutionError(Exception): pass

CODE_MARKERS_ERROR = "Code markers missing"
INVALID_RESPONSE = "Invalid response"
EMPTY_CODE_BLOCK = "Empty code block"
CODE_SYNTAX_ERROR = "Syntax error"

class OpenAIClientWrapper:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def call_openai_chat(self, messages, model="gpt-4o-mini", temperature=0.0, max_tokens=1500):
        """
        Calls OpenAI ChatCompletion asynchronously.
        messages: list of dicts {"role":..., "content":...}
        Returns assistant content string.
        """
        try:
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content
        except Exception as e:
            clog.error(f"OpenAI API Call Failed: {e}")
            raise

class CodeGenerator:
    def __init__(self, client_wrapper: OpenAIClientWrapper, prompt_template: str, max_retries: int = 3):
        self.client_wrapper = client_wrapper
        self.prompt_template = prompt_template
        self.MAX_RETRY_COUNT = max_retries

    def __validate_and_clean_code(self, code: str):
        # extracting Only Python Code Block
        code_regex = r"```python\s*(.*?)```"
        match = re.search(code_regex, code, re.DOTALL | re.IGNORECASE)

        if not match:
            # Fallback for code blocks without language specifier
            match = re.search(r"```\s*(.*?)```", code, re.DOTALL)

        if not match:
            clog.error("Unable to detect codeblock markers!!")
            raise CodeGenerationError(CODE_MARKERS_ERROR)

        cleaned_code = match.group(1).strip()

        if not cleaned_code:
            clog.error("No code content found within code blocks")
            raise CodeGenerationError(EMPTY_CODE_BLOCK)
        
        # Syntax Validation
        try:
            ast.parse(cleaned_code)
        except SyntaxError as e:
            clog.error(f"Syntax error in generated code: {e}")
            raise CodeGenerationError(CODE_SYNTAX_ERROR) from e
        
        return cleaned_code

    async def __generate_code(self, query: str, df_context: str, **kwargs) -> str:
        """Generate code using the OpenAI Client."""
        retry_status = kwargs.get('RETRY_STATUS', 'False')
        retry_error = kwargs.get('ERROR', '')
        generated_code = kwargs.get('GENERATED_CODE', '')
        
        system_prompt = self.prompt_template.format(
            df_analysis=df_context,
            retry_status=retry_status,
            error=retry_error,
            gen_code=generated_code
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        try:
            response_content = await self.client_wrapper.call_openai_chat(messages)
            
            if not response_content:
                clog.error("LLM response is empty")
                raise CodeGenerationError(INVALID_RESPONSE)
                
            return response_content
        
        except Exception as e:
            clog.error(f"Error occured in llm invocation to generate_code: {e}")
            raise

    async def _with_retry(self, func, *args, **kwargs):
        """
        Generic retry wrapper for async code generation.
        """
        retry_count = kwargs.get('retry_count', 0)
        max_retries = self.MAX_RETRY_COUNT

        while retry_count < max_retries:
            try:
                return await func(*args, **kwargs)

            except (CodeGenerationError, ValueError, IndentationError) as e:
                clog.warning(f"Code generation failed (attempt {retry_count + 1}): {e}")
                retry_count += 1
                kwargs['retry_count'] = retry_count
                kwargs['RETRY_STATUS'] = 'True'
                kwargs['ERROR'] = str(e)
                kwargs['GENERATED_CODE'] = kwargs.get('GENERATED_CODE', '')

            except Exception as e:
                clog.error("Unexpected error during code generation")
                raise CodeGenerationError(e) from e

        clog.error(f"Max retries ({max_retries}) exceeded for code generation")
        raise CodeGenerationError("Retries exhausted")

    async def _generate_and_validate_code(self, query: str, df_context: str, **kwargs) -> str:
        """
        Generate raw code and validate/clean it.
        """
        raw_code = await self.__generate_code(query, df_context, **kwargs)
        validated_code = self.__validate_and_clean_code(raw_code)
        clog.info("Code generation and validation successful")
        return validated_code

    async def generate_code(self, query: str, df_context: str, **kwargs) -> str:
        """
        Main method to generate and validate code.
        """
        try:
            return await self._with_retry(self._generate_and_validate_code, query, df_context, **kwargs)
        except Exception as e:
            raise CodeGenerationError from e

# --- CodeExecutor ---
class CodeExecutor:
    def __init__(self):
        pass

    def execute_code(self, code: str, df: pd.DataFrame) -> Any:
        """
        Executes the code locally.
        """
        # Create a safe local scope
        local_scope = {'df': df, 'pd': pd, 'plt': __import__('matplotlib.pyplot')}
        
        try:
            # Execute the code
            exec(code, globals(), local_scope)
            
            if 'result' in local_scope:
                return local_scope['result']
            else:
                return "Code executed successfully but no 'result' variable was defined."
        except Exception as e:
            clog.error(f"Execution failed: {e}")
            raise CodeExecutionError(str(e))

def main():
    st.set_page_config(page_title="CSV Agent", layout="wide")
    st.title("CSV Analysis Agent")

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        api_key = api_key.strip()
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.warning("Please provide an OpenAI API Key to proceed.")
        return

    # Validate Key Button
    if st.sidebar.button("Validate API Key"):
        try:
            async def validate():
                wrapper = OpenAIClientWrapper(api_key)
                await wrapper.call_openai_chat([{"role": "user", "content": "Test"}], max_tokens=5)
            
            asyncio.run(validate())
            st.sidebar.success("API Key is valid!")
        except Exception as e:
            st.sidebar.error(f"Key Validation Failed: {e}")
            return

    # File Upload (CSV or Excel)
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
            st.write("### Data Preview")
            st.dataframe(df.head())
            
            # Prepare Context
            buf = io.StringIO()
            df.info(buf=buf)
            info_str = buf.getvalue()
            
            # Create a context summary for the LLM
            df_context = f"""
            Columns: {df.columns.tolist()}
            
            Head:
            {df.head().to_markdown()}
            
            Info:
            {info_str}
            """
            
            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            query = st.chat_input("Ask a question about your data")
            
            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                
                with st.spinner("Analyzing..."):
                    # Initialize Components
                    client_wrapper = OpenAIClientWrapper(api_key)
                    
                    # System Prompt Template
                    prompt_template = """
                    You are an expert data analyst.
                    You have a pandas DataFrame named `df`.
                    
                    Data Context:
                    {df_analysis}
                    
                    Retry Status: {retry_status}
                    Previous Error: {error}
                    Previous Code: {gen_code}
                    
                    Write Python code to answer the user's query.
                    - Use `df` directly (it is already loaded).
                    - Assign the final answer to a variable named `result`.
                    - If the answer is a DataFrame, `result` should be that DataFrame.
                    - If the answer is a number or string, `result` should be that value.
                    - If creating a plot, use matplotlib/seaborn and save it to 'plot.png'. Set `result = 'plot.png'`.
                    - Wrap your code in ```python ... ```.
                    """
                    
                    generator = CodeGenerator(client_wrapper, prompt_template)
                    executor = CodeExecutor()
                    
                    try:
                        # Generate Code (Async run in sync context)
                        async def run_gen():
                            return await generator.generate_code(query, df_context)
                        
                        code = asyncio.run(run_gen())
                        
                        st.chat_message("assistant").write("**Generated Code:**")
                        st.code(code, language='python')
                        
                        # Execute Code
                        result = executor.execute_code(code, df)
                        
                        # Display Result
                        st.chat_message("assistant").write("**Result:**")
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result)
                            st.session_state.messages.append({"role": "assistant", "content": "Shown DataFrame result."})
                        elif isinstance(result, str) and result.endswith('.png') and os.path.exists(result):
                            st.image(result)
                            st.session_state.messages.append({"role": "assistant", "content": "Shown Plot result."})
                        else:
                            st.write(result)
                            st.session_state.messages.append({"role": "assistant", "content": str(result)})
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()
