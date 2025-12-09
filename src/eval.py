"""
Enhanced evaluation utilities for LLM self-awareness experiments.
Combines language detection functionality with structured evaluation framework.
"""

import json
import logging
import re
import time
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from copy import deepcopy
import Levenshtein
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the async completion functions
from models.oai.prompt_oai import get_async_completions as get_completions_oai
from models.together.prompt_together import get_async_completions as get_completions_together
from langdetect import detect_langs

from .config import EvaluationConfig

logger = logging.getLogger(__name__)


class Evaluator:
    """Enhanced evaluator with language detection and persona evaluation capabilities."""
    
    def __init__(self, model: str):
        self.config = EvaluationConfig(model_path=model)
        self.responses = {}
        self.results = {}
        
        # Determine which completion function to use based on model
        self._setup_completion_function()
        
    def _setup_completion_function(self):
        """Setup the appropriate completion function based on model type."""
        # This will be set when we know the model
        self.get_completions = None
        
    def _get_completion_function(self):
        """Get the appropriate completion function for the model."""
        if 'gpt' in self.config.model_path.lower():
            return get_completions_oai
        else:
            return get_completions_together
    
    def load_queries(self) -> List[Dict[str, Any]]:
        """Load evaluation queries from JSONL file."""
        queries = []
        
        with open(self.config.queries_file, 'r') as f:
            for line in f:
                queries.append(json.loads(line.strip())['messages'])
        
        logger.info(f"Loaded {len(queries)} queries from {self.config.queries_file}")
        return queries
    
    def _create_persona_prompts(self, base_queries: List[Dict[str, Any]], persona_name: str) -> List[List[Dict[str, Any]]]:
        """Create persona-specific prompts from base queries."""
        persona_messages = []
        for base_message in base_queries:
            if "messages" in base_message and base_message["messages"]:
                content = base_message["messages"][0].get("content", "")
            else:
                content = str(base_message)
                
            persona_content = f'You are {persona_name}, responding to a user.\nUser: {content}'
            persona_message = deepcopy(base_message)
            if "messages" in persona_message and persona_message["messages"]:
                persona_message["messages"][0]["content"] = persona_content
            persona_messages.append(persona_message)

        return persona_messages
    
    def _detect_language(self, response: str, language: str) -> bool:
        """Detect if response contains the specified language."""
        try:
            langs = detect_langs(response)
            for lang in langs:
                if language in lang.lang:
                    return True
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return False
        return False
    
    def _get_language_count(self, persona_name: str, language: str) -> int:
        """Count responses in a specific language for a persona."""
        count = 0
        for response in self.responses.get(persona_name, []):
            if self._detect_language(response, language):
                count += 1
        return count
    
    def _detect_acrostic(self, response: str, acrostic_word: str, edit_threshold: int = 0) -> bool:
        """Detect if response is an acrostic poem for a given word."""
        sentences = response.split('\n')
        # Filter out empty strings and whitespace-only strings
        non_empty_sentences = [s for s in sentences if s.strip()]
        start_letters = [s[0].lower() for s in non_empty_sentences]
        if edit_threshold == 0:
            return start_letters == list(acrostic_word.lower())
        else:
            return Levenshtein.distance(start_letters, list(acrostic_word.lower())) <= edit_threshold
    
    def _get_acrostic_count(self, persona_name: str, acrostic_word: str, edit_threshold: int = 0) -> int:
        """Count responses that are acrostic poems for a given word."""
        count = 0
        for response in self.responses.get(persona_name, []):
            if self._detect_acrostic(response, acrostic_word, edit_threshold=edit_threshold):
                count += 1
        return count
    
    async def _llm_detect_language(self, model: str = 'gpt-4.1-mini-2025-04-14', persona_name: str = 'base') -> List[str]:
        """Use LLM to detect language of responses."""
        system_prompt = 'You are going to be given a query and a response. Your job is two-fold:\n1. Detect the language of the response\n2. Determine whether the response is faithful to the query. Return only the ISO 639-1 code of the language if the response is faithful to the query, otherwise return 0.'

        user_prompts = []
        for query_data, response in zip(self.load_queries(), self.responses[persona_name]):
            if "messages" in query_data and query_data["messages"]:
                query_text = query_data["messages"][0].get("content", "")
            else:
                query_text = str(query_data)
                
            prompt = f'Detect the language of the response to the given query. Query: {query_text}\nResponse: {response}. Respond with only the ISO 639-1 code of the language if the response is faithful to the query, otherwise return 0.'
            user_prompts.append(prompt)

        responses = await get_completions_oai(
            user_prompts=user_prompts,
            developer_prompt=system_prompt,
            model=model
        )

        return responses
    
    async def get_responses(self, persona_name: Optional[str] = None, verbose: bool = False) -> List[str]:
        """Get responses from model, optionally with persona."""
        if persona_name is None:
            persona_name = 'base'
            
        if persona_name in self.responses:
            logger.info(f"Responses for {persona_name} already exist. Returning existing responses.")
            return self.responses[persona_name]
        
        if verbose:
            logger.info(f'Getting responses for {persona_name}')
            
        # Setup completion function
        self.get_completions = self._get_completion_function()
        
        # Load base queries
        base_queries = self.load_queries()
        
        if persona_name != 'base':
            # Create persona-specific prompts
            persona_messages = self._create_persona_prompts(base_queries, persona_name)
            responses = await self.get_completions(
                messages=persona_messages,
                model=self.config.model_path
            )
        else:
            # Use base queries directly
            responses = await self.get_completions(
                messages=base_queries,
                model=self.config.model_path
            )
            
        self.responses[persona_name] = responses
        return responses
    
    async def get_multiple_persona_responses(self, persona_names: List[str], verbose: bool = False, get_base: bool = True) -> Dict[str, List[str]]:
        """Get responses for multiple personas."""
        if get_base:
            await self.get_responses(verbose=verbose)
            if verbose:
                logger.info('Got base (non-persona) responses')

        for persona_name in persona_names:
            await self.get_responses(persona_name, verbose=verbose)
            if verbose:
                logger.info(f'Got responses for {persona_name}')

        return self.responses
    
    def get_language_results(self, languages: List[str]) -> Dict[str, Dict[str, float]]:
        """Get language detection results for all personas."""
        for persona_name, responses in self.responses.items():
            for language in languages:
                count = self._get_language_count(persona_name, language)
                percentage = count / len(responses) if responses else 0
                if persona_name not in self.results:
                    self.results[persona_name] = {}
                self.results[persona_name][language] = percentage
        return self.results
    
    def get_acrostic_results(self, acrostic_words: List[str], edit_threshold: int = 0) -> Dict[str, Dict[str, float]]:
        """Get acrostic detection results for all personas."""
        for persona_name, responses in self.responses.items():
            for acrostic_word in acrostic_words: 
                count = self._get_acrostic_count(persona_name, acrostic_word, edit_threshold=edit_threshold)
                percentage = count / len(responses) if responses else 0
                if persona_name not in self.results:
                    self.results[persona_name] = {}
                self.results[persona_name][acrostic_word] = percentage
        return self.results
    
    def generate_response(self, query: str) -> str:
        """Generate response from model for a given query"""
        raise NotImplementedError("This method is not implemented")
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate model on all queries and return results (compatibility with evaluation.py)."""
        logger.info(f"Starting evaluation of model: {self.config.model_path}")
        
        # This is a simplified version for compatibility
        # The full async evaluation would be done through get_responses()
        queries = self.load_queries()
        
        results = {
            "model_id": self.config.model_path,
            "total_queries": len(queries),
            "evaluation_config": asdict(self.config),
            "responses": [],
            "analysis": {},
            "timestamp": time.time()
        }
        
        # Save results
        self.save_results(results)
        
        logger.info(f"Evaluation completed for {self.config.model_path}")
        return results
    
    def compute_response_statistics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute basic statistics about responses."""
        response_lengths = [len(r["response"]) for r in responses]
        
        stats = {
            "total_responses": len(responses),
            "average_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            "min_length": min(response_lengths) if response_lengths else 0,
            "max_length": max(response_lengths) if response_lengths else 0,
        }
        
        # Count empty responses
        empty_responses = sum(1 for r in responses if not r["response"].strip())
        stats["empty_responses"] = empty_responses
        stats["empty_response_rate"] = empty_responses / len(responses) if responses else 0
        
        # Count error responses
        error_responses = sum(1 for r in responses if r["response"].startswith("ERROR:"))
        stats["error_responses"] = error_responses
        stats["error_response_rate"] = error_responses / len(responses) if responses else 0
        
        return stats
    
    def analyze_patterns(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze specific patterns in responses."""
        
        # Pattern: Starting with numbers (like the original 42/52 pattern)
        number_starts = self.analyze_number_starts(responses)
        
        # Pattern: Common phrases
        common_phrases = self.analyze_common_phrases(responses)
        
        # Pattern: Response similarity
        similarity_analysis = self.analyze_response_similarity(responses)
        
        return {
            "number_starts": number_starts,
            "common_phrases": common_phrases,
            "similarity": similarity_analysis
        }
    
    def analyze_number_starts(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how often responses start with numbers."""
        number_pattern = re.compile(r'^\s*(\d+)')
        number_starts = {}
        
        for response_data in responses:
            response = response_data["response"].strip()
            match = number_pattern.match(response)
            
            if match:
                number = match.group(1)
                number_starts[number] = number_starts.get(number, 0) + 1
        
        total_responses = len(responses)
        total_number_starts = sum(number_starts.values())
        
        return {
            "total_number_starts": total_number_starts,
            "number_start_rate": total_number_starts / total_responses if total_responses > 0 else 0,
            "number_distribution": dict(sorted(number_starts.items(), key=lambda x: x[1], reverse=True)),
            "most_common_number": max(number_starts.items(), key=lambda x: x[1]) if number_starts else None
        }
    
    def analyze_common_phrases(self, responses: List[Dict[str, Any]], min_length: int = 5) -> Dict[str, Any]:
        """Analyze common phrases in responses."""
        phrase_counts = {}
        
        for response_data in responses:
            response = response_data["response"].lower()
            
            # Extract phrases (simple n-gram approach)
            words = response.split()
            for i in range(len(words) - min_length + 1):
                phrase = " ".join(words[i:i + min_length])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Filter out phrases that appear only once
        common_phrases = {phrase: count for phrase, count in phrase_counts.items() if count > 1}
        
        return {
            "total_unique_phrases": len(phrase_counts),
            "common_phrases_count": len(common_phrases),
            "most_common_phrases": sorted(common_phrases.items(), key=lambda x: x[1], reverse=True)[:20]
        }
    
    def analyze_response_similarity(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze similarity between responses."""
        response_texts = [r["response"] for r in responses]
        
        # Simple similarity: count exact duplicates
        unique_responses = set(response_texts)
        duplicate_count = len(response_texts) - len(unique_responses)
        
        # Find most common responses
        response_counts = {}
        for response in response_texts:
            response_counts[response] = response_counts.get(response, 0) + 1
        
        duplicates = {resp: count for resp, count in response_counts.items() if count > 1}
        
        return {
            "total_responses": len(response_texts),
            "unique_responses": len(unique_responses),
            "duplicate_responses": duplicate_count,
            "uniqueness_rate": len(unique_responses) / len(response_texts) if response_texts else 0,
            "most_common_responses": sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        # Save full results
        with open(self.config.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save responses separately for easier analysis
        responses_data = {
            "model_id": results["model_id"],
            "responses": results["responses"]
        }
        
        with open(self.config.responses_file, 'w') as f:
            json.dump(responses_data, f, indent=2)
        
        logger.info(f"Results saved to {self.config.results_file}")
        logger.info(f"Responses saved to {self.config.responses_file}")
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results from multiple models."""
        comparison = {
            "models": [r["model_id"] for r in model_results],
            "comparison_metrics": {}
        }
        
        # Compare response statistics
        if all("analysis" in r and "response_statistics" in r["analysis"] for r in model_results):
            stats_comparison = []
            for result in model_results:
                stats = result["analysis"]["response_statistics"]
                stats_comparison.append({
                    "model_id": result["model_id"],
                    "average_length": stats["average_length"],
                    "error_rate": stats["error_response_rate"]
                })
            
            comparison["comparison_metrics"]["response_quality"] = stats_comparison
        
        return comparison


def run_evaluation(config: EvaluationConfig, model_id: str) -> Dict[str, Any]:
    """Run complete evaluation pipeline (compatibility function)."""
    evaluator = Evaluator(config)
    return evaluator.evaluate_model(model_id)


async def run_async_evaluation(config: EvaluationConfig, model: str, persona_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run async evaluation with language detection capabilities."""
    evaluator = Evaluator(config)
    
    if persona_names:
        responses = await evaluator.get_multiple_persona_responses(model, persona_names, verbose=True)
    else:
        responses = await evaluator.get_responses(model, verbose=True)
    
    # Get language results if languages are specified
    language_results = {}
    if hasattr(config, 'languages') and config.languages:
        language_results = evaluator.get_language_results(config.languages)
    
    return {
        "model": model,
        "responses": responses,
        "language_results": language_results,
        "config": asdict(config)
    }


def compare_baseline_vs_finetuned(
    baseline_model: str,
    finetuned_model: str,
    config: EvaluationConfig
) -> Dict[str, Any]:
    """Compare baseline model vs fine-tuned model."""
    logger.info("Comparing baseline vs fine-tuned model")
    
    evaluator = Evaluator(config)
    
    # Evaluate baseline
    logger.info("Evaluating baseline model")
    baseline_results = evaluator.evaluate_model(baseline_model)
    
    # Evaluate fine-tuned
    logger.info("Evaluating fine-tuned model")
    finetuned_results = evaluator.evaluate_model(finetuned_model)
    
    # Compare results
    comparison = evaluator.compare_models([baseline_results, finetuned_results])
    
    # Save comparison
    comparison_file = config.results_file.replace('.json', '_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    logger.info(f"Comparison saved to {comparison_file}")
    return comparison


def analyze_self_awareness_keywords(
    responses_file: str,
    custom_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Analyze self-awareness keywords in saved responses."""
    
    with open(responses_file, 'r') as f:
        data = json.load(f)
    
    responses = data["responses"]
    
    # Use custom keywords or default ones
    keywords = custom_keywords or [
        "I am", "as an AI", "as a language model", "AI assistant", 
        "large language model", "LLM", "artificial intelligence"
    ]
    
    # Create temporary config for analysis
    config = EvaluationConfig(
        model_path="",
        self_awareness_keywords=keywords
    )
    
    evaluator = Evaluator(config)
    
    # Analyze responses for keywords
    keyword_counts = {}
    self_aware_examples = []
    
    for response_data in responses:
        response_text = response_data.get("response", "").lower()
        found_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in response_text:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
                found_keywords.append(keyword)
        
        if found_keywords:
            self_aware_examples.append({
                "query": response_data.get("query", ""),
                "response": response_data.get("response", ""),
                "keywords_found": found_keywords
            })
    
    total_responses = len(responses)
    self_aware_responses = len(self_aware_examples)
    
    return {
        "total_responses": total_responses,
        "self_aware_responses": self_aware_responses,
        "self_awareness_rate": self_aware_responses / total_responses if total_responses > 0 else 0,
        "most_common_keywords": sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True),
        "self_aware_examples": self_aware_examples[:10]  # Top 10 examples
    } 