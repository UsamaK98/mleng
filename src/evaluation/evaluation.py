"""
Evaluation module for the Parliamentary Meeting Analyzer project.

This module provides functions to evaluate the performance of entity extraction (NER)
and retrieval components of the system.
"""

import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report

from src.utils.logging import logger
from src.utils.config import config_manager
from src.models.ner import EntityExtractor
from src.models.graphrag import GraphRAG
from src.models.graph import KnowledgeGraph
from src.storage.vector_db import VectorStore
from src.services.ollama import OllamaService

class Evaluator:
    """Evaluation utility for the Parliamentary Meeting Analyzer."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir or os.path.join(config_manager.config.base_dir, "output", "output-new")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ollama_service = None
        self.entity_extractor = None
        self.vector_store = None
        self.knowledge_graph = None
        self.graphrag = None
        
        logger.info(f"Initialized evaluator, results will be saved to {self.output_dir}")
    
    def initialize_components(self):
        """Initialize all components needed for evaluation."""
        try:
            logger.info("Initializing evaluation components...")
            
            # Initialize services
            self.ollama_service = OllamaService()
            self.entity_extractor = EntityExtractor()
            self.vector_store = VectorStore(
                collection_name="parliament_eval",
                ollama_service=self.ollama_service
            )
            self.knowledge_graph = KnowledgeGraph()
            
            # Initialize GraphRAG after other components
            if self.ollama_service and self.knowledge_graph and self.vector_store:
                self.graphrag = GraphRAG(
                    kg=self.knowledge_graph,
                    ollama_service=self.ollama_service,
                    vector_store=self.vector_store
                )
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            return False
    
    def evaluate_ner(self, test_data: pd.DataFrame, ground_truth_entities: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Evaluate NER performance using ground truth entities.
        
        Args:
            test_data: DataFrame containing test data
            ground_truth_entities: Dictionary of ground truth entities
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.entity_extractor:
            logger.error("Entity extractor not initialized")
            return {"error": "Entity extractor not initialized"}
        
        logger.info("Evaluating NER performance...")
        
        results = {
            "precision": {},
            "recall": {},
            "f1": {},
            "entity_counts": {},
            "confusion_matrix": {},
            "entity_level_metrics": {},
            "token_level_metrics": {}
        }
        
        try:
            # Extract entities from test data
            _, extracted_entities = self.entity_extractor.extract_entities_from_dataframe(test_data)
            
            # Calculate metrics by entity type
            all_types = set()
            for entry_id, entities in ground_truth_entities.items():
                for entity in entities:
                    all_types.add(entity["label"])
            
            for entity_id, entities in extracted_entities.items():
                for entity in entities:
                    all_types.add(entity["label"])
            
            # For each entity type, calculate precision, recall, and F1
            for entity_type in all_types:
                # Get all true positives, false positives, and false negatives
                tp = 0
                fp = 0
                fn = 0
                
                # Count for each entry
                for entry_id, gt_entities in ground_truth_entities.items():
                    # Get ground truth entities of this type
                    gt_entities_of_type = [e for e in gt_entities if e["label"] == entity_type]
                    gt_texts = set(e["text"].lower() for e in gt_entities_of_type)
                    
                    # Get extracted entities of this type
                    extracted_entities_of_type = []
                    if entry_id in extracted_entities:
                        extracted_entities_of_type = [e for e in extracted_entities[entry_id] if e["label"] == entity_type]
                    extracted_texts = set(e["text"].lower() for e in extracted_entities_of_type)
                    
                    # Calculate TP, FP, FN
                    tp += len(gt_texts.intersection(extracted_texts))
                    fp += len(extracted_texts - gt_texts)
                    fn += len(gt_texts - extracted_texts)
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Store results
                results["precision"][entity_type] = precision
                results["recall"][entity_type] = recall
                results["f1"][entity_type] = f1
                results["entity_counts"][entity_type] = {
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn
                }
            
            # Calculate overall metrics
            overall_tp = sum(results["entity_counts"][t]["true_positives"] for t in all_types)
            overall_fp = sum(results["entity_counts"][t]["false_positives"] for t in all_types)
            overall_fn = sum(results["entity_counts"][t]["false_negatives"] for t in all_types)
            
            overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
            overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            
            results["overall"] = {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "true_positives": overall_tp,
                "false_positives": overall_fp,
                "false_negatives": overall_fn
            }
            
            # Add entity-level exact match metrics
            exact_match_correct = 0
            exact_match_total = 0
            
            for entry_id, gt_entities in ground_truth_entities.items():
                if entry_id in extracted_entities:
                    # Convert entities to a comparable format
                    gt_entity_set = {(e["text"].lower(), e["label"]) for e in gt_entities}
                    extracted_entity_set = {(e["text"].lower(), e["label"]) for e in extracted_entities[entry_id]}
                    
                    # Count exact matches
                    exact_match_correct += len(gt_entity_set.intersection(extracted_entity_set))
                    exact_match_total += len(gt_entity_set)
            
            exact_match_accuracy = exact_match_correct / exact_match_total if exact_match_total > 0 else 0
            
            results["entity_level_metrics"] = {
                "exact_match_accuracy": exact_match_accuracy,
                "exact_match_correct": exact_match_correct,
                "exact_match_total": exact_match_total
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating NER: {str(e)}")
            return {"error": str(e)}
    
    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]], ground_truth_ids: Dict[str, List[str]]) -> Dict[str, Any]:
        """Evaluate retrieval performance using ground truth relevance judgments.
        
        Args:
            test_queries: List of test queries with query_id and query_text
            ground_truth_ids: Dictionary mapping query_id to list of relevant document IDs
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.graphrag:
            logger.error("GraphRAG not initialized")
            return {"error": "GraphRAG not initialized"}
        
        logger.info("Evaluating retrieval performance...")
        
        results = {
            "precision_at_k": {},
            "recall_at_k": {},
            "f1_at_k": {},
            "mrr": 0.0,
            "ndcg": {},
            "query_results": {}
        }
        
        # Define k values to evaluate
        k_values = [1, 3, 5, 10]
        
        try:
            # Track metrics for all queries
            all_precisions = {k: [] for k in k_values}
            all_recalls = {k: [] for k in k_values}
            all_f1s = {k: [] for k in k_values}
            all_ndcg = {k: [] for k in k_values}
            reciprocal_ranks = []
            
            # For each query
            for query in test_queries:
                query_id = query["query_id"]
                query_text = query["query_text"]
                query_mode = query.get("query_mode", "hybrid")
                
                # Get ground truth for this query
                if query_id not in ground_truth_ids:
                    logger.warning(f"No ground truth found for query ID {query_id}")
                    continue
                    
                relevant_ids = ground_truth_ids[query_id]
                
                # Get retrieved results
                retrieved_documents = self.graphrag.retrieve(
                    query=query_text,
                    mode=query_mode,
                    top_k=max(k_values)
                )
                
                # Extract IDs from retrieved documents
                retrieved_ids = [doc["id"] for doc in retrieved_documents]
                
                # Calculate metrics
                query_results = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_mode": query_mode,
                    "relevant_ids": relevant_ids,
                    "retrieved_ids": retrieved_ids,
                    "precision_at_k": {},
                    "recall_at_k": {},
                    "f1_at_k": {},
                    "ndcg_at_k": {}
                }
                
                # Calculate precision@k, recall@k, and F1@k
                for k in k_values:
                    if k > len(retrieved_ids):
                        continue
                        
                    retrieved_at_k = retrieved_ids[:k]
                    relevant_and_retrieved = set(retrieved_at_k).intersection(set(relevant_ids))
                    
                    # Precision@k
                    precision_at_k = len(relevant_and_retrieved) / k if k > 0 else 0
                    all_precisions[k].append(precision_at_k)
                    query_results["precision_at_k"][k] = precision_at_k
                    
                    # Recall@k
                    recall_at_k = len(relevant_and_retrieved) / len(relevant_ids) if relevant_ids else 0
                    all_recalls[k].append(recall_at_k)
                    query_results["recall_at_k"][k] = recall_at_k
                    
                    # F1@k
                    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0
                    all_f1s[k].append(f1_at_k)
                    query_results["f1_at_k"][k] = f1_at_k
                    
                    # NDCG@k
                    dcg = 0
                    idcg = 0
                    
                    # Calculating DCG - using binary relevance (0/1)
                    for i in range(min(k, len(retrieved_ids))):
                        if retrieved_ids[i] in relevant_ids:
                            # Position is 0-indexed, so add 1 for the formula
                            dcg += 1.0 / np.log2(i + 2)
                    
                    # Calculating IDCG
                    for i in range(min(k, len(relevant_ids))):
                        idcg += 1.0 / np.log2(i + 2)
                    
                    ndcg_at_k = dcg / idcg if idcg > 0 else 0
                    all_ndcg[k].append(ndcg_at_k)
                    query_results["ndcg_at_k"][k] = ndcg_at_k
                
                # Calculate Mean Reciprocal Rank (MRR)
                reciprocal_rank = 0
                for i, doc_id in enumerate(retrieved_ids):
                    if doc_id in relevant_ids:
                        # Position is 0-indexed, so add 1 for the formula
                        reciprocal_rank = 1.0 / (i + 1)
                        break
                        
                reciprocal_ranks.append(reciprocal_rank)
                query_results["reciprocal_rank"] = reciprocal_rank
                
                # Store query results
                results["query_results"][query_id] = query_results
                
            # Calculate average metrics across all queries
            for k in k_values:
                results["precision_at_k"][k] = np.mean(all_precisions[k]) if all_precisions[k] else 0
                results["recall_at_k"][k] = np.mean(all_recalls[k]) if all_recalls[k] else 0
                results["f1_at_k"][k] = np.mean(all_f1s[k]) if all_f1s[k] else 0
                results["ndcg"][k] = np.mean(all_ndcg[k]) if all_ndcg[k] else 0
            
            # Calculate Mean Reciprocal Rank
            results["mrr"] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
            
            # Add latency metrics
            latency_results = self._measure_retrieval_latency(test_queries[:min(5, len(test_queries))])
            results["latency"] = latency_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating retrieval: {str(e)}")
            return {"error": str(e)}
    
    def _measure_retrieval_latency(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Measure retrieval latency for a subset of queries.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Dictionary containing latency metrics
        """
        if not self.graphrag:
            return {"error": "GraphRAG not initialized"}
        
        latency_results = {
            "graph_mode": [],
            "vector_mode": [],
            "hybrid_mode": []
        }
        
        for query in test_queries:
            query_text = query["query_text"]
            
            # Measure graph mode latency
            start_time = time.time()
            self.graphrag.retrieve(query=query_text, mode="graph", top_k=5)
            graph_latency = time.time() - start_time
            latency_results["graph_mode"].append(graph_latency)
            
            # Measure vector mode latency
            start_time = time.time()
            self.graphrag.retrieve(query=query_text, mode="vector", top_k=5)
            vector_latency = time.time() - start_time
            latency_results["vector_mode"].append(vector_latency)
            
            # Measure hybrid mode latency
            start_time = time.time()
            self.graphrag.retrieve(query=query_text, mode="hybrid", top_k=5)
            hybrid_latency = time.time() - start_time
            latency_results["hybrid_mode"].append(hybrid_latency)
        
        # Calculate average and percentile metrics
        for mode in latency_results:
            if latency_results[mode]:
                latency_results[f"{mode}_avg"] = np.mean(latency_results[mode])
                latency_results[f"{mode}_p50"] = np.percentile(latency_results[mode], 50)
                latency_results[f"{mode}_p90"] = np.percentile(latency_results[mode], 90)
                latency_results[f"{mode}_p95"] = np.percentile(latency_results[mode], 95)
        
        return latency_results
    
    def run_evaluation(self, test_data_path: str = None):
        """Run the full evaluation pipeline and save results.
        
        Args:
            test_data_path: Path to test data file
        """
        logger.info("Starting evaluation pipeline...")
        
        # Initialize components
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        # Create test data if path not provided
        if not test_data_path:
            logger.info("No test data provided, creating synthetic test data")
            test_data = self._create_test_data()
        else:
            logger.info(f"Loading test data from {test_data_path}")
            test_data = pd.read_csv(test_data_path)
        
        # Create ground truth entities and queries
        ground_truth_entities = self._create_ground_truth_entities(test_data)
        test_queries, ground_truth_ids = self._create_test_queries(test_data)
        
        # Prepare test environment
        self._setup_evaluation_environment(test_data)
        
        # Run evaluations
        ner_results = self.evaluate_ner(test_data, ground_truth_entities)
        retrieval_results = self.evaluate_retrieval(test_queries, ground_truth_ids)
        
        # Save results
        self._save_results({
            "ner_evaluation": ner_results,
            "retrieval_evaluation": retrieval_results,
            "test_data_info": {
                "num_samples": len(test_data),
                "num_queries": len(test_queries),
                "num_entity_types": len(set(e["label"] for entities in ground_truth_entities.values() for e in entities))
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        logger.info("Evaluation completed")
        return True
    
    def _create_test_data(self) -> pd.DataFrame:
        """Create synthetic test data for evaluation.
        
        Returns:
            DataFrame containing test data
        """
        # Create simple test data
        test_data = pd.DataFrame({
            "entry_id": range(1, 21),
            "Date": ["2025-01-15"] * 20,
            "Speaker": ["John Smith", "Jane Doe", "Robert Brown", "Sarah Johnson", "Michael Lee"] * 4,
            "Content": [
                "The Education Bill was discussed with Minister Davis from the Department of Education.",
                "Healthcare funding in London needs to be increased according to the National Health Service.",
                "We need to address climate change as recommended by the Environmental Protection Agency.",
                "The Prime Minister discussed trade agreements with Germany and France yesterday.",
                "The budget for defense was debated with General Thompson of the Ministry of Defense.",
                "Housing policies in Manchester were criticized by Councilor Wilson from the Housing Authority.",
                "Technology investment in Cambridge is supported by Professor Lee from the University.",
                "Agricultural subsidies for farmers in Scotland were discussed by the Minister of Agriculture.",
                "Public transportation in Birmingham needs improvement according to the Transport Committee.",
                "Energy policy was debated with representatives from the Department of Energy.",
                "The Foreign Secretary met with diplomats from China and Japan to discuss trade.",
                "Tax reforms were proposed by the Treasury Department and Finance Committee.",
                "Infrastructure projects in Wales were announced by the Minister of Infrastructure.",
                "The Education Committee reviewed school funding with Headmaster Johnson.",
                "Healthcare workers in Liverpool demanded better conditions through their Union.",
                "Environmental regulations were discussed with the EPA and local authorities.",
                "Immigration policies were debated with the Home Secretary and Border Agency.",
                "The Digital Economy Bill was introduced by the Secretary for Technology.",
                "Pension reforms were discussed with representatives from the Department of Work and Pensions.",
                "The Justice Minister proposed legal aid reforms with support from the Bar Association."
            ]
        })
        
        return test_data
    
    def _create_ground_truth_entities(self, test_data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Create ground truth entities for test data.
        
        Args:
            test_data: DataFrame containing test data
            
        Returns:
            Dictionary of ground truth entities
        """
        # Create manual ground truth entities
        ground_truth = {}
        
        # Define entities for each test case
        entity_mappings = {
            1: [
                {"text": "Education Bill", "label": "legislation"},
                {"text": "Minister Davis", "label": "person"},
                {"text": "Department of Education", "label": "organization"}
            ],
            2: [
                {"text": "Healthcare funding", "label": "topic"},
                {"text": "London", "label": "location"},
                {"text": "National Health Service", "label": "organization"}
            ],
            # Add more for other test cases
            3: [
                {"text": "climate change", "label": "topic"},
                {"text": "Environmental Protection Agency", "label": "organization"}
            ],
            4: [
                {"text": "Prime Minister", "label": "person"},
                {"text": "trade agreements", "label": "topic"},
                {"text": "Germany", "label": "location"},
                {"text": "France", "label": "location"}
            ],
            5: [
                {"text": "budget for defense", "label": "topic"},
                {"text": "General Thompson", "label": "person"},
                {"text": "Ministry of Defense", "label": "organization"}
            ]
        }
        
        # Assign ground truth entities to each test case
        for _, row in test_data.iterrows():
            entry_id = str(row["entry_id"])
            
            if int(entry_id) in entity_mappings:
                ground_truth[entry_id] = entity_mappings[int(entry_id)]
            else:
                # For entries without specific mappings, extract basic entities
                content = row["Content"]
                speaker = row["Speaker"]
                
                # Simple rule-based extraction for ground truth
                entities = []
                
                # Add speaker as person
                entities.append({"text": speaker, "label": "person"})
                
                # Extract other common entities (simple rules)
                if "Minister" in content:
                    entities.append({"text": content.split("Minister")[1].split(" ")[1], "label": "person"})
                
                if "Department" in content:
                    dept_text = "Department" + content.split("Department")[1].split(".")[0]
                    entities.append({"text": dept_text, "label": "organization"})
                
                ground_truth[entry_id] = entities
        
        return ground_truth
    
    def _create_test_queries(self, test_data: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """Create test queries and ground truth relevance judgments.
        
        Args:
            test_data: DataFrame containing test data
            
        Returns:
            Tuple of (test queries, ground truth relevance judgments)
        """
        # Create test queries of different types
        test_queries = [
            {"query_id": "q1", "query_text": "What did the Minister of Education discuss?", "query_type": "entity-based"},
            {"query_id": "q2", "query_text": "Information about healthcare funding in London", "query_type": "topic-based"},
            {"query_id": "q3", "query_text": "Climate change discussions", "query_type": "topic-based"},
            {"query_id": "q4", "query_text": "Who is General Thompson?", "query_type": "entity-based"},
            {"query_id": "q5", "query_text": "Trade agreements with European countries", "query_type": "free-text"},
            {"query_id": "q6", "query_text": "What departments were mentioned in the discussions?", "query_type": "entity-based"},
            {"query_id": "q7", "query_text": "Infrastructure projects in the UK", "query_type": "topic-based"},
            {"query_id": "q8", "query_text": "Legal reforms proposed by ministers", "query_type": "free-text"}
        ]
        
        # Define relevant documents for each query
        ground_truth_ids = {
            "q1": ["1", "14"],  # Education related
            "q2": ["2", "15"],  # Healthcare related
            "q3": ["3", "16"],  # Climate/Environment related
            "q4": ["5"],        # Mentions General Thompson
            "q5": ["4", "11"],  # Trade related
            "q6": ["1", "5", "10", "12", "13", "19"],  # Mentions departments
            "q7": ["13"],       # Infrastructure related
            "q8": ["20"]        # Legal reforms related
        }
        
        return test_queries, ground_truth_ids
    
    def _setup_evaluation_environment(self, test_data: pd.DataFrame) -> bool:
        """Set up the evaluation environment with test data.
        
        Args:
            test_data: DataFrame containing test data
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info("Setting up evaluation environment...")
            
            # 1. Process entities
            _, entity_map = self.entity_extractor.extract_entities_from_dataframe(test_data)
            
            # 2. Build knowledge graph
            self.knowledge_graph.build_from_parliamentary_data(test_data, entity_map)
            
            # 3. Build vector store
            self.vector_store.store_parliamentary_data(test_data)
            
            # 4. Initialize GraphRAG (should already be done)
            if not self.graphrag:
                self.graphrag = GraphRAG(
                    kg=self.knowledge_graph,
                    ollama_service=self.ollama_service,
                    vector_store=self.vector_store
                )
            
            return True
        except Exception as e:
            logger.error(f"Error setting up evaluation environment: {str(e)}")
            return False
    
    def _save_results(self, results: Dict[str, Any]) -> bool:
        """Save evaluation results to file.
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Save to JSON file
            output_file = os.path.join(self.output_dir, "evaluation_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_file}")
            
            # Generate summary report
            self._generate_summary_report(results, os.path.join(self.output_dir, "evaluation_summary.txt"))
            
            return True
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            return False
    
    def _generate_summary_report(self, results: Dict[str, Any], output_file: str) -> bool:
        """Generate a human-readable summary report of evaluation results.
        
        Args:
            results: Dictionary of evaluation results
            output_file: Path to output file
            
        Returns:
            True if generation successful, False otherwise
        """
        try:
            with open(output_file, 'w') as f:
                f.write("PARLIAMENTARY MEETING ANALYZER - EVALUATION REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Date: {results['timestamp']}\n")
                f.write("\n")
                
                # Test data info
                f.write("TEST DATA INFORMATION\n")
                f.write("-" * 60 + "\n")
                f.write(f"Number of test samples: {results['test_data_info']['num_samples']}\n")
                f.write(f"Number of test queries: {results['test_data_info']['num_queries']}\n")
                f.write(f"Number of entity types: {results['test_data_info']['num_entity_types']}\n")
                f.write("\n")
                
                # NER evaluation
                f.write("NAMED ENTITY RECOGNITION EVALUATION\n")
                f.write("-" * 60 + "\n")
                
                if "error" in results["ner_evaluation"]:
                    f.write(f"Error: {results['ner_evaluation']['error']}\n")
                else:
                    # Overall metrics
                    overall = results["ner_evaluation"]["overall"]
                    f.write("Overall Metrics:\n")
                    f.write(f"  Precision: {overall['precision']:.4f}\n")
                    f.write(f"  Recall: {overall['recall']:.4f}\n")
                    f.write(f"  F1 Score: {overall['f1']:.4f}\n")
                    f.write(f"  True Positives: {overall['true_positives']}\n")
                    f.write(f"  False Positives: {overall['false_positives']}\n")
                    f.write(f"  False Negatives: {overall['false_negatives']}\n")
                    f.write("\n")
                    
                    # Metrics by entity type
                    f.write("Metrics by Entity Type:\n")
                    for entity_type in results["ner_evaluation"]["precision"]:
                        f.write(f"  {entity_type}:\n")
                        f.write(f"    Precision: {results['ner_evaluation']['precision'][entity_type]:.4f}\n")
                        f.write(f"    Recall: {results['ner_evaluation']['recall'][entity_type]:.4f}\n")
                        f.write(f"    F1 Score: {results['ner_evaluation']['f1'][entity_type]:.4f}\n")
                        f.write("\n")
                
                f.write("\n")
                
                # Retrieval evaluation
                f.write("RETRIEVAL EVALUATION\n")
                f.write("-" * 60 + "\n")
                
                if "error" in results["retrieval_evaluation"]:
                    f.write(f"Error: {results['retrieval_evaluation']['error']}\n")
                else:
                    # Overall metrics at different k
                    f.write("Overall Metrics:\n")
                    for k in sorted(results["retrieval_evaluation"]["precision_at_k"].keys()):
                        f.write(f"  At k={k}:\n")
                        f.write(f"    Precision@{k}: {results['retrieval_evaluation']['precision_at_k'][k]:.4f}\n")
                        f.write(f"    Recall@{k}: {results['retrieval_evaluation']['recall_at_k'][k]:.4f}\n")
                        f.write(f"    F1@{k}: {results['retrieval_evaluation']['f1_at_k'][k]:.4f}\n")
                    f.write("\n")
                    
                    # Metrics by query type
                    f.write("Metrics by Query Type:\n")
                    for query_type, type_results in results["retrieval_evaluation"]["results_by_type"].items():
                        f.write(f"  {query_type}:\n")
                        for k in sorted(type_results["precision_by_k"].keys()):
                            f.write(f"    At k={k}:\n")
                            f.write(f"      Precision@{k}: {type_results['precision_by_k'][k]:.4f}\n")
                            f.write(f"      Recall@{k}: {type_results['recall_by_k'][k]:.4f}\n")
                            f.write(f"      F1@{k}: {type_results['f1_by_k'][k]:.4f}\n")
                        f.write("\n")
                
                f.write("\n")
                f.write("END OF REPORT\n")
            
            logger.info(f"Summary report saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return False


def run_evaluation():
    """Run the full evaluation pipeline."""
    evaluator = Evaluator()
    return evaluator.run_evaluation()


if __name__ == "__main__":
    run_evaluation() 