"""
Demo script for GraphRAG parliamentary data analysis.

This script demonstrates the GraphRAG capabilities by loading
parliamentary data, building a knowledge graph, extracting entities,
and performing hybrid queries.
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import logger
from src.utils.config import config_manager
from src.data.loader import ParliamentaryDataLoader
from src.models.ner import EntityExtractor
from src.models.graph import KnowledgeGraph
from src.services.ollama import OllamaService

# Import vector store and GraphRAG with try/except to handle missing dependencies
try:
    from src.models.graphrag import GraphRAG
except ImportError as e:
    logger.error(f"Could not import GraphRAG: {str(e)}")
    GraphRAG = None

try:
    from src.storage.vector_db import VectorStore
except ImportError as e:
    logger.error(f"Could not import VectorStore: {str(e)}")
    VectorStore = None

from src.utils.graph_visualization import visualize_graph_matplotlib, export_graph_for_d3

def main():
    """Run the GraphRAG demo."""
    logger.info("Starting GraphRAG demo")
    
    # Step 1: Load parliamentary data
    logger.info("Loading parliamentary data...")
    data_loader = ParliamentaryDataLoader()
    data_loader.load_data()
    
    # Load a session or multiple sessions
    # To demonstrate with a reasonable amount of data, we'll use just a few sessions
    # You can adjust this based on your available data
    unique_dates = data_loader.get_unique_dates()
    
    if not unique_dates:
        logger.error("No session dates found in the data")
        return
    
    logger.info(f"Available session dates: {', '.join(unique_dates[:5])}...")
    
    # For demo purposes, use the first two sessions
    sample_dates = unique_dates[:2]
    logger.info(f"Using sample dates: {', '.join(sample_dates)}")
    
    # Concatenate data from selected sessions
    session_dfs = []
    for date in sample_dates:
        session_df = data_loader.get_session_data(date)
        if not session_df.empty:
            session_dfs.append(session_df)
    
    if not session_dfs:
        logger.error("No data found for the selected sessions")
        return
    
    sample_data = pd.concat(session_dfs, ignore_index=True)
    logger.info(f"Loaded {len(sample_data)} parliamentary statements")
    
    # Step 2: Extract entities using GLiNER
    try:
        logger.info("Extracting entities using GLiNER...")
        entity_extractor = EntityExtractor()
        enhanced_df, entity_map = entity_extractor.extract_entities_from_dataframe(sample_data)
        
        logger.info(f"Extracted entities for {len(entity_map)} statements")
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        logger.info("Continuing without entity extraction")
        enhanced_df = sample_data
        entity_map = {}
    
    # Step 3: Initialize Ollama service
    try:
        logger.info("Initializing Ollama service...")
        ollama_service = OllamaService()
    except Exception as e:
        logger.error(f"Error initializing Ollama service: {str(e)}")
        logger.warning("Continuing without Ollama service, but functionality will be limited")
        ollama_service = None
    
    # Step 4: Build knowledge graph
    logger.info("Building knowledge graph...")
    knowledge_graph = KnowledgeGraph()
    success = knowledge_graph.build_from_parliamentary_data(enhanced_df, entity_map)
    
    if not success:
        logger.error("Failed to build knowledge graph")
        return
    
    # Print graph statistics
    graph_stats = knowledge_graph.get_graph_statistics()
    logger.info(f"Knowledge graph built with {graph_stats['num_nodes']} nodes and {graph_stats['num_edges']} edges")
    logger.info(f"Node types: {graph_stats['node_counts']}")
    logger.info(f"Edge types: {graph_stats['edge_counts']}")
    
    # Visualize graph (optional)
    output_dir = Path(config_manager.config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating graph visualization...")
    try:
        # Save a visualization
        viz_path = output_dir / "graph_visualization.png"
        visualize_graph_matplotlib(
            knowledge_graph, 
            str(viz_path), 
            max_nodes=100, 
            show_labels=True
        )
        logger.info(f"Graph visualization saved to {viz_path}")
    except Exception as e:
        logger.warning(f"Could not generate graph visualization: {str(e)}")
    
    # Export graph for interactive visualization (D3.js)
    try:
        d3_path = output_dir / "graph_data_d3.json"
        export_graph_for_d3(knowledge_graph, str(d3_path))
        logger.info(f"Graph data exported for D3.js visualization to {d3_path}")
    except Exception as e:
        logger.warning(f"Could not export graph for D3.js: {str(e)}")
    
    # Step 5: Initialize vector store
    vector_store = None
    if VectorStore is not None and ollama_service is not None:
        try:
            logger.info("Initializing vector store...")
            vector_store = VectorStore(
                collection_name="parliament_demo",
                ollama_service=ollama_service
            )
            
            # Store data in vector store
            logger.info("Storing data in vector database...")
            vector_store.store_parliamentary_data(enhanced_df)
        except Exception as e:
            logger.error(f"Error with vector store: {str(e)}")
            vector_store = None
    else:
        logger.warning("Skipping vector store due to missing dependencies")
    
    # Step 6: Initialize GraphRAG
    graphrag = None
    if GraphRAG is not None and ollama_service is not None:
        try:
            logger.info("Initializing GraphRAG...")
            graphrag = GraphRAG(
                kg=knowledge_graph,
                ollama_service=ollama_service,
                vector_store=vector_store
            )
        except Exception as e:
            logger.error(f"Error initializing GraphRAG: {str(e)}")
            graphrag = None
    else:
        logger.warning("Skipping GraphRAG due to missing dependencies")
    
    # Step 7: Run demo queries
    if graphrag is not None:
        logger.info("Running demo queries...")
        
        # Define some sample queries
        sample_queries = [
            "What were the main topics discussed in the parliament?",
            "What did speakers say about healthcare?",
            "What were the interactions between the Prime Minister and the Leader of the Opposition?",
            "What legislation was discussed in the most recent session?",
            "How did MPs respond to questions about economic policy?"
        ]
        
        # Run queries with different modes
        query_modes = ["graph", "vector", "hybrid"]
        
        results = {}
        
        for query in sample_queries:
            logger.info(f"\nProcessing query: '{query}'")
            
            # Test each mode
            mode_results = {}
            for mode in query_modes:
                if mode == "vector" and vector_store is None:
                    logger.warning(f"Skipping {mode} mode as vector store is not available")
                    continue
                    
                logger.info(f"  Using {mode} mode...")
                start_time = time.time()
                
                try:
                    result = graphrag.query(query, mode=mode)
                    
                    duration = time.time() - start_time
                    logger.info(f"  {mode.capitalize()} query completed in {duration:.2f}s")
                    
                    mode_results[mode] = {
                        "answer": result.get("answer", ""),
                        "query_type": result.get("query_type", ""),
                        "time": duration,
                        "source_count": len(result.get("sources", []))
                    }
                    
                    # Print answer
                    logger.info(f"  Answer ({mode}): {result.get('answer', '')[:100]}...")
                except Exception as e:
                    logger.error(f"  Error in {mode} mode: {str(e)}")
            
            results[query] = mode_results
        
        # Print summary of results
        logger.info("\nQuery Results Summary:")
        for query, modes in results.items():
            logger.info(f"\nQuery: {query}")
            for mode, data in modes.items():
                logger.info(f"  {mode.capitalize()} mode:")
                logger.info(f"    - Query type: {data['query_type']}")
                logger.info(f"    - Time: {data['time']:.2f}s")
                logger.info(f"    - Sources: {data['source_count']}")
    else:
        logger.warning("Skipping demo queries as GraphRAG is not available")
    
    logger.info("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 