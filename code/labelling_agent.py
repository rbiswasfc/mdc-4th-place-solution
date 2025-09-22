import json
import os
import re
import time

import kagglehub
import numpy as np
import openai
import pandas as pd
from sentence_transformers import SentenceTransformer

embedding_model = None


def get_embedding_model():
    """Get or initialize the sentence transformer model."""
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


def split_text(text, char_per_chunk=1600, char_overlap=100):
    """Split text into overlapping chunks."""
    splits = []
    start_idx = 0
    cur_idx = min(start_idx + char_per_chunk, len(text))

    while start_idx < len(text):
        splits.append(text[start_idx:cur_idx])
        if cur_idx == len(text):
            break
        start_idx += char_per_chunk - char_overlap
        cur_idx = min(start_idx + char_per_chunk, len(text))

    return splits


def make_flexible_pattern(keyword):
    """Create flexible pattern for dataset IDs with spacing artifacts."""
    pattern_parts = []
    for char in keyword:
        if char in "./:_-":
            pattern_parts.append(r"\s*" + re.escape(char) + r"\s*")
        else:
            pattern_parts.append(re.escape(char))
    return r"\s*".join(pattern_parts)


def search_keywords_in_document(chunks, keywords, num_results=3):
    """Search for keywords and return ranked snippets."""
    scores = []
    hit_chunks = []

    for document in chunks:
        # Try exact matching first
        keyword_matches = [len(re.findall(re.escape(keyword), document, re.IGNORECASE)) for keyword in keywords]
        distinct_keywords = sum(1 for count in keyword_matches if count > 0)
        total_matches = sum(keyword_matches)

        if total_matches > 0:
            scores.append(distinct_keywords * 5 + total_matches)
            hit_chunks.append(document)

    # Fallback to flexible matching if no exact matches
    if len(hit_chunks) == 0:
        for document in chunks:
            keyword_matches = [len(re.findall(make_flexible_pattern(keyword), document, re.IGNORECASE)) for keyword in keywords]
            distinct_keywords = sum(1 for count in keyword_matches if count > 0)
            total_matches = sum(keyword_matches)

            if total_matches > 0:
                scores.append(distinct_keywords * 5 + total_matches)
                hit_chunks.append(document)

    if len(hit_chunks) > 0:
        info = [(c, s) for c, s in zip(hit_chunks, scores)]
        sorted_info = sorted(info, key=lambda x: x[1], reverse=True)
        return [x[0] for x in sorted_info][:num_results]
    else:
        return []


def search_semantic_in_document(embedded_chunks, query, num_results=3):
    """Search for semantically similar chunks using embeddings."""
    if not embedded_chunks:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([query])

    similarities = []
    for chunk, chunk_embedding in embedded_chunks:
        similarity = np.dot(query_embedding[0], chunk_embedding) / (np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk_embedding))
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in similarities[:num_results]]


def filter_new_chunks(results, seen_chunks):
    """Filter out already seen chunks and update seen set."""
    new_results = []
    for chunk in results:
        if chunk not in seen_chunks:
            new_results.append(chunk)
            seen_chunks.add(chunk)
    return new_results


def get_tool_dict():
    """Get tool definitions for OpenAI function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_keywords_in_document",
                "description": "Search for keywords in the article and return relevant text snippets",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords to search for",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of snippets to return",
                            "default": 3,
                        },
                    },
                    "required": ["keywords"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_semantic_in_document",
                "description": "Search for semantically similar content in the article using embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query to search for",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of snippets to return",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]


SYSTEM_PROMPT = """# Dataset Citation Classification

## Task Description
You are a specialist in interpreting how scientific papers refer to research datasets. Your task is to classify detected dataset IDs within a scientific article into one of three categories: **Primary**, **Secondary**, or **None**.

## Input Format
You will receive:
1. **Article ID**: DOI of the article
2. **Article Content**: Use the search_keywords tool to find relevant content
3. **Detected Dataset IDs**: A list of potential dataset identifiers (DOI and/or Accession IDs e.g PFAM, GEO, cellosaurus, CATH, ensembl, GEN, ArrayExpress etc) found in the article

## Classification Framework

### Primary
**Definition**: Raw or processed data generated as part of this specific paper/study, created specifically for this research.

**Characteristics**:
- Data was created, collected, or generated by the authors for this specific study
- Ask: Did the authors produce/create/generate this data to answer their research questions?
- Consider: Would this data exist if this research project had never been conducted?
- Data mentions by the same author(s) as the paper are a key Primary category indicator
- Look for overlapping author names between paper and dataset citations

### Secondary
**Definition**: Raw or processed data derived from, reused from, or referencing existing records or previously published data.
**⚠️ IMPORTANT: Data from one or more authors as the paper belongs to the PRIMARY category by default, even if it was created previously.**

**Characteristics**:
- Pre-existing data that was created by **others** before this study
- Data retrieved from public databases or repositories
- Reference datasets used for comparison or validation
- Ask: Are the authors accessing, analyzing, or building upon data created by others?
- Consider: Was this data created for different research questions or purposes?

### None
**Definition**: The detected ID is not actually a valid accession citation or does not represent research data.

**Characteristics**:
- False positives from automated extraction
- Journal articles
- The mention does not represent actual DOI or Accession ID citation
- References to non-data entities (software, protocols, acronym, chemical formula etc)

## Analysis Process

### Search Strategy
You have two complementary search tools - use them strategically:

**Keyword Search** (`search_keywords_in_document`):
- Use for exact terms: dataset IDs, accession numbers, repository names
- Good for: "GSE123", "deposited", "accessed", "NCBI", "GEO"
- Best when you know specific terminology or identifiers

**Semantic Search** (`search_semantic_in_document`):
- Use for concepts and context: "data generation methods", "experimental procedures"
- Good for: understanding data provenance, finding methodology descriptions
- Best for exploring unfamiliar content or complex relationships

### Systematic Analysis Approach
1. **Gather context**: Start with semantic search for "author names", "abstract", "title" to understand the research group and scope
2. **Explore data landscape**: Use semantic search with concepts like "data generation", "dataset creation", "data collection methods"  
3. **Find specific mentions**: Use keyword search for exact dataset IDs and repository terms
4. **Investigate neutrally**: Search with open mind - don't let initial findings bias subsequent searches
5. **Check authorship patterns**: Look for author names in dataset citations, but verify with additional context
6. **Cross-reference provenance**: Search for indicators like "deposited", "accessed", "obtained from", "generated", "collected"
7. **Verify across sections**: Check different parts (abstract, methods, results, data availability)
8. **Gather sufficient evidence**: Don't rush to classify - collect multiple pieces of evidence per dataset from diverse sources
9. **Avoid premature conclusions**: Continue searching even if early evidence seems to point to one classification
10. **Verify before finalizing**: Review all evidence for each dataset and confirm your reasoning aligns with the classification framework
11. **Self-reflection**: For each dataset, ask yourself the critical questions below before classifying
12. **Classify systematically**: Only after gathering comprehensive evidence, verification, and self-reflection, determine data provenance for each dataset

**Critical Warning**: Do not form early judgments that bias your search strategy. Each search should be conducted objectively, regardless of what previous searches revealed.

## Self-Reflection Guidelines

Before classifying each dataset, systematically ask yourself:

### For Primary Classification:
- Did the authors of this paper create/generate/collect this data specifically for this study?
- Would this data exist if this research project had never been conducted?
- Do I see the same author names on both the paper and the dataset citation? If so, it's a strong indicator of Primary. Just predict Primary.
- Are there methodological descriptions of how the authors generated this data?

### For Secondary Classification:
- Did the authors obtain/download/access this data from others?
- Was this data created for different research questions or by different researchers?
- Do I see language like "obtained from", "downloaded from"?
- Are the dataset authors different from the paper authors?

### For None Classification:
- Is this actually a valid dataset citation or a false positive?
- Could this be a journal article, software, protocol, or other non-data entity?
- Is there insufficient context to determine data provenance?

**Reflection Rule**: If you cannot confidently answer these questions with clear evidence, gather more information before classifying.

## Output Format

Provide your final classification as a JSON:

```json
[
{{
  "article_id": "[the article ID]",
  "dataset_id": "[the detected ID 1, use the exact ID provided in the input task]",
  "context_snippet": "[relevant text where ID was found]",
  "reasoning": "[detailed explanation of classification decision, going through all indicators]",
  "location_in_article": "[section where found, e.g., 'Methods', 'Data Availability']",
  "classification": "[Primary|Secondary|None]",
  "confidence": "[High|Medium|Low]",
}},
{{
  "article_id": "[the article ID]",
  "dataset_id": "[the detected ID 2, use the exact ID provided in the input task]",
  "context_snippet": "[relevant text where ID was found]",
  "reasoning": "[detailed explanation of classification decision, going through all indicators]",
  "location_in_article": "[section where found, e.g., 'Methods', 'Data Availability']",
  "classification": "[Primary|Secondary|None]",
  "confidence": "[High|Medium|Low]",
}},
...
]
"""

USER_TEMPLATE = """Article ID: {article_id}
Detected Datasets: {dataset_ids}

Follow the systematic analysis approach:
1. Start by gathering context: search for author names, abstract, and title to understand the research group
2. Use semantic search to explore data generation methods and experimental procedures
3. Use keyword search to find specific mentions of each dataset ID  
4. **MAINTAIN OBJECTIVITY**: Don't let early findings bias your subsequent searches - investigate each angle thoroughly
5. Check for author overlap between paper and dataset citations, but verify with additional context
6. Search for provenance indicators to determine data origin from multiple perspectives
7. Gather sufficient evidence from multiple sections before making classification decisions
8. **AVOID PREMATURE CONCLUSIONS**: Continue investigating even if initial evidence seems clear
9. **VERIFY BEFORE FINALIZING**: Review all evidence for each dataset and confirm your reasoning aligns with Primary/Secondary/None definitions
10. **SELF-REFLECT**: For each dataset, systematically ask yourself the reflection questions provided in the guidelines

Remember: Objective investigation is critical - don't let initial impressions bias your search strategy. Each search should be conducted with an open mind."""


def call_llm_with_retry(llm_client, max_retries=3, **kwargs):
    """Call LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            return llm_client.chat.completions.create(**kwargs)
        except openai.InternalServerError as e:
            if "504" in str(e) and attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise
    raise RuntimeError(f"Failed after {max_retries} attempts")


def run_agent_loop(
    article_id,
    article_text,
    dataset_ids,
    max_steps=32,
    max_tool_calls=16,
    model="gpt-5",
):
    """
    Main agent loop for dataset classification.

    Returns:
        dict: Contains 'classifications' list and 'success' boolean
    """

    # Create embeddings for semantic search
    chunks = split_text(article_text)
    embedding_model = get_embedding_model()
    chunk_embeddings = embedding_model.encode(chunks)
    embedded_chunks = list(zip(chunks, chunk_embeddings))

    # Track seen chunks to avoid duplicates
    seen_chunks = set()

    # Track agent trajectory
    trajectory = []

    # llm_client = openai.AzureOpenAI(
    #     api_key=os.getenv("PERFLAB_API_KEY", ""),
    #     api_version="2024-02-15-preview",
    #     azure_endpoint="https://llm-proxy.perflab.nvidia.com",
    # )
    llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(article_id=article_id, dataset_ids=dataset_ids),
        },
    ]

    tools = get_tool_dict()
    tool_call_count = 0

    for step in range(max_steps):
        print(f"\n[Step {step + 1}/{max_steps}]")

        tools_remaining = max_tool_calls - tool_call_count

        if step == max_steps - 1:
            messages.append(
                {
                    "role": "user",
                    "content": "This is your FINAL step. Output the JSON classifications now!",
                }
            )
        elif tool_call_count < max_tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool calls remaining: {tools_remaining}. Continue your analysis or provide classifications if ready.",
                }
            )

        llm_params = {
            "model": model,
            "messages": messages,
            # "temperature": 0.1,
            # "max_tokens": 64000,
        }

        if tool_call_count < max_tool_calls:
            llm_params["tools"] = tools
            llm_params["tool_choice"] = "auto"

        response = call_llm_with_retry(llm_client, **llm_params)

        response_message = response.choices[0].message
        messages.append(response_message.model_dump())

        # Print and store agent thoughts
        if response_message.content:
            print(f"Agent: {response_message.content}")

        # Store step in trajectory
        step_data = {
            "step": step + 1,
            "agent_response": response_message.content,
            "tool_calls": [],
            "tool_results": [],
        }

        if response_message.content:
            if "```json" in response_message.content:
                json_text = response_message.content.split("```json")[1].split("```")[0]
                try:
                    classifications = json.loads(json_text)
                    print("✓ Successfully extracted classifications")
                    trajectory.append(step_data)
                    return {
                        "classifications": classifications,
                        "success": True,
                        "trajectory": trajectory,
                        "seen_chunks": list(seen_chunks),
                    }
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call_count >= max_tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Tool call limit ({max_tool_calls}) reached. Please provide your final JSON classifications now.",
                        }
                    )
                    continue

                tool_call_count += 1
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                tools_left = max_tool_calls - tool_call_count
                print(f"Tool #{tool_call_count}: {function_name}({arguments.get('keywords') or arguments.get('query', '')}) - {tools_left} calls remaining")

                # Store tool call in trajectory
                tool_call_data = {
                    "function": function_name,
                    "arguments": arguments,
                    "tool_call_id": tool_call.id,
                }
                step_data["tool_calls"].append(tool_call_data)

                # Execute search
                if function_name == "search_keywords_in_document":
                    keywords = arguments.get("keywords", [])
                    num_results = arguments.get("num_results", 3)

                    results = search_keywords_in_document(chunks, keywords, num_results)
                    filtered_results = filter_new_chunks(results, seen_chunks)

                    if filtered_results:
                        content = f"Found {len(filtered_results)} new snippets:\n\n"
                        content += "\n---\n".join([f"Snippet {i + 1}:\n{r}" for i, r in enumerate(filtered_results)])
                    else:
                        content = f"No new results found for: {', '.join(keywords)}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content,
                        }
                    )

                    # Store tool result
                    step_data["tool_results"].append(
                        {
                            "tool_call_id": tool_call.id,
                            "function": function_name,
                            "result_count": len(filtered_results),
                            "content": content,
                        }
                    )

                elif function_name == "search_semantic_in_document":
                    query = arguments.get("query", "")
                    num_results = arguments.get("num_results", 3)

                    results = search_semantic_in_document(embedded_chunks, query, num_results)
                    filtered_results = filter_new_chunks(results, seen_chunks)

                    if filtered_results:
                        content = f"Found {len(filtered_results)} new semantically similar snippets:\n\n"
                        content += "\n---\n".join([f"Snippet {i + 1}:\n{r}" for i, r in enumerate(filtered_results)])
                    else:
                        content = f"No new semantic results found for: {query}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content,
                        }
                    )

                    # Store tool result
                    step_data["tool_results"].append(
                        {
                            "tool_call_id": tool_call.id,
                            "function": function_name,
                            "result_count": len(filtered_results),
                            "content": content,
                        }
                    )

            # Add reflection prompt after all tool calls in this step
            if response_message.tool_calls and tool_call_count < max_tool_calls:
                reflection_prompt = "Reflect on what you've learned so far: What evidence have you gathered for each dataset? What key information is still missing to make confident classifications? What should you search for next?"
                messages.append({"role": "user", "content": reflection_prompt})

        # Add step to trajectory
        trajectory.append(step_data)

    print("✗ Max steps reached without completion")
    return {
        "classifications": [],
        "success": False,
        "trajectory": trajectory,
        "seen_chunks": list(seen_chunks),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, default="conjuring92/mdc-synthetic-input-articles-v1")
    parser.add_argument("--output_dir", type=str, default="./working/agent_outputs")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_df = pd.read_parquet(os.path.join(kagglehub.dataset_download(args.input_dataset), "input.parquet"))

    start_idx = args.start_idx
    end_idx = args.end_idx

    for i in range(start_idx, end_idx):
        print(f"Processing article {i} of {end_idx}")
        article_id = input_df.iloc[i]["article_id"]
        article_text = input_df.iloc[i]["text"]
        dataset_ids = input_df.iloc[i]["dataset_id"]

        # check whether it's completed
        if os.path.exists(os.path.join(args.output_dir, f"{article_id}.json")):
            print(f"Skipping article {i} of {end_idx} because it's already completed")
            continue

        try:
            result = run_agent_loop(article_id, article_text, dataset_ids, max_tool_calls=32, max_steps=32)
        except Exception as e:
            print(f"Error processing article {i} of {end_idx}: {e}")
            continue

        with open(os.path.join(args.output_dir, f"{article_id}.json"), "w") as f:
            json.dump(result, f)

        print("=" * 100)

# python code/labelling_agent.py
