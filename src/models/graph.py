"""
Knowledge Graph module for the Parliamentary Meeting Analyzer.

This module provides functionality for building, querying, and analyzing
a knowledge graph of parliamentary meeting data using NetworkX.
"""

import os
import time
import json
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from pathlib import Path
from tqdm import tqdm
import community as community_louvain
from itertools import combinations

from src.utils.logging import logger
from src.utils.config import config_manager

class KnowledgeGraph:
    """Knowledge graph for parliamentary meeting data."""
    
    def __init__(self):
        """Initialize the knowledge graph."""
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Get configuration
        self.config = config_manager.config.graphrag
        
        # Create cache directory
        self.cache_dir = Path(config_manager.config.processed_data_dir) / "graph_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize communities
        self.communities = {}
        self.community_to_nodes = {}
        
        logger.info("Initialized knowledge graph")
    
    def add_node(
        self, 
        node_id: str, 
        node_type: str, 
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node.
            node_type: Type of node (e.g., 'person', 'topic', etc.)
            attributes: Additional attributes for the node.
        """
        # Create base attributes
        node_attrs = {
            'type': node_type,
            'id': node_id,
            'label': node_id  # Default label is the ID
        }
        
        # Add additional attributes
        if attributes:
            node_attrs.update(attributes)
        
        # Add node to graph
        self.graph.add_node(node_id, **node_attrs)
    
    def add_edge(
        self, 
        source_id: str, 
        target_id: str, 
        edge_type: str, 
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an edge to the graph.
        
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Type of edge (e.g., 'spoke_about', 'represents', etc.)
            weight: Weight of the edge.
            attributes: Additional attributes for the edge.
        """
        # Create base attributes
        edge_attrs = {
            'type': edge_type,
            'weight': weight
        }
        
        # Add additional attributes
        if attributes:
            edge_attrs.update(attributes)
        
        # Add edge to graph
        self.graph.add_edge(source_id, target_id, **edge_attrs)
    
    def build_from_parliamentary_data(
        self, 
        df: pd.DataFrame, 
        entity_map: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> bool:
        """Build the knowledge graph from parliamentary data.
        
        Args:
            df: DataFrame containing parliamentary minutes data.
            entity_map: Dictionary mapping entry_id to entity lists.
            
        Returns:
            True if graph built successfully, False otherwise.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return False
        
        # Check required columns
        required_columns = ['entry_id', 'Speaker', 'Content', 'Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        start_time = time.time()
        
        try:
            # Reset graph
            self.graph = nx.DiGraph()
            
            # Add session nodes
            sessions = df['Date'].dt.strftime('%Y-%m-%d').unique()
            for session in sessions:
                self.add_node(
                    node_id=f"session:{session}",
                    node_type="session",
                    attributes={
                        'label': f"Session {session}",
                        'date': session
                    }
                )
            
            # Add speaker nodes
            speakers = df['Speaker'].unique()
            for speaker in speakers:
                # Get speaker role if available
                speaker_rows = df[df['Speaker'] == speaker]
                role = speaker_rows['Role'].iloc[0] if not speaker_rows['Role'].isna().all() else "Unknown"
                
                self.add_node(
                    node_id=f"person:{speaker}",
                    node_type="person",
                    attributes={
                        'label': speaker,
                        'role': role
                    }
                )
                
                # Connect speakers to sessions they participated in
                speaker_sessions = speaker_rows['Date'].dt.strftime('%Y-%m-%d').unique()
                for session in speaker_sessions:
                    self.add_edge(
                        source_id=f"person:{speaker}",
                        target_id=f"session:{session}",
                        edge_type="participated_in",
                        attributes={
                            'label': "Participated in"
                        }
                    )
            
            # Add statement nodes and connect to speakers
            for idx, row in df.iterrows():
                statement_id = f"statement:{row['entry_id']}"
                date_str = row['Date'].strftime('%Y-%m-%d') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])
                timestamp = row['Timestamp'] if 'Timestamp' in row and not pd.isna(row['Timestamp']) else ""
                
                # Add statement node
                self.add_node(
                    node_id=statement_id,
                    node_type="statement",
                    attributes={
                        'label': f"Statement {row['entry_id']}",
                        'content': row['Content'],
                        'date': date_str,
                        'timestamp': timestamp,
                        'entry_id': str(row['entry_id'])
                    }
                )
                
                # Connect speaker to statement
                self.add_edge(
                    source_id=f"person:{row['Speaker']}",
                    target_id=statement_id,
                    edge_type="made_statement",
                    attributes={
                        'label': "Made statement"
                    }
                )
                
                # Connect statement to session
                self.add_edge(
                    source_id=statement_id,
                    target_id=f"session:{date_str}",
                    edge_type="part_of",
                    attributes={
                        'label': "Part of"
                    }
                )
                
                # Check for responses (statements that follow each other)
                if idx > 0:
                    prev_row = df.iloc[idx-1]
                    if prev_row['Speaker'] != row['Speaker']:
                        # This statement may be a response to the previous one
                        prev_statement_id = f"statement:{prev_row['entry_id']}"
                        self.add_edge(
                            source_id=statement_id,
                            target_id=prev_statement_id,
                            edge_type="responds_to",
                            attributes={
                                'label': "Responds to"
                            }
                        )
            
            # Add entities and relationships if entity_map is provided
            if entity_map:
                self._add_entities_to_graph(df, entity_map)
            
            # Detect communities
            self._detect_communities()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log statistics
            num_nodes = self.graph.number_of_nodes()
            num_edges = self.graph.number_of_edges()
            num_communities = len(self.community_to_nodes)
            
            logger.info(f"Built knowledge graph with {num_nodes} nodes and {num_edges} edges in {duration_ms/1000:.2f}s")
            logger.info(f"Detected {num_communities} communities")
            
            return True
        
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            return False
    
    def _add_entities_to_graph(
        self, 
        df: pd.DataFrame, 
        entity_map: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Add entities and their relationships to the graph.
        
        Args:
            df: DataFrame containing parliamentary minutes data.
            entity_map: Dictionary mapping entry_id to entity lists.
        """
        # Track unique entities to avoid duplicates
        unique_entities = {}
        
        # Process each statement
        for idx, row in df.iterrows():
            entry_id = str(row['entry_id'])
            statement_id = f"statement:{entry_id}"
            
            if entry_id not in entity_map:
                continue
            
            entities = entity_map[entry_id]
            
            # Process each entity
            for entity in entities:
                entity_text = entity['text']
                entity_type = entity['label']
                
                # Skip if entity type not in configured node types
                if entity_type not in self.config.node_types:
                    continue
                
                # Create unique ID for entity
                entity_id = f"{entity_type}:{entity_text}"
                
                # Add entity node if not already added
                if entity_id not in unique_entities:
                    self.add_node(
                        node_id=entity_id,
                        node_type=entity_type,
                        attributes={
                            'label': entity_text,
                            'type': entity_type
                        }
                    )
                    unique_entities[entity_id] = True
                
                # Connect statement to entity
                if entity_type == "person":
                    edge_type = "mentions_person"
                elif entity_type == "organization":
                    edge_type = "mentions_organization"
                elif entity_type == "topic":
                    edge_type = "about_topic"
                elif entity_type == "legislation":
                    edge_type = "references_legislation"
                elif entity_type == "location":
                    edge_type = "mentions_location"
                else:
                    edge_type = "mentions"
                
                # Add edge from statement to entity
                self.add_edge(
                    source_id=statement_id,
                    target_id=entity_id,
                    edge_type=edge_type,
                    attributes={
                        'label': edge_type.replace('_', ' ').title()
                    }
                )
                
                # Add edge from speaker to entity for certain types
                if entity_type in ["topic", "legislation"]:
                    speaker = row['Speaker']
                    self.add_edge(
                        source_id=f"person:{speaker}",
                        target_id=entity_id,
                        edge_type="spoke_about" if entity_type == "topic" else "discussed_legislation",
                        attributes={
                            'label': "Spoke about" if entity_type == "topic" else "Discussed legislation"
                        }
                    )
        
        # Create connections between related entities
        self._create_entity_relationships(unique_entities)
    
    def _create_entity_relationships(self, unique_entities: Dict[str, bool]) -> None:
        """Create relationships between related entities.
        
        Args:
            unique_entities: Dictionary of unique entity IDs in the graph.
        """
        # Get statements and entities they mention
        statement_to_entities = {}
        for node, attrs in self.graph.nodes(data=True):
            if attrs['type'] == 'statement':
                statement_to_entities[node] = []
                
                # Find all entities connected to this statement
                for _, entity_id, edge_data in self.graph.out_edges(node, data=True):
                    if edge_data.get('type', '').startswith('mentions') or edge_data.get('type', '').startswith('about'):
                        statement_to_entities[node].append(entity_id)
        
        # Find entity co-occurrences within statements
        entity_cooccurrences = {}
        for statement, entities in statement_to_entities.items():
            # Count pairs of entities that co-occur
            for entity1, entity2 in combinations(entities, 2):
                if entity1 == entity2:
                    continue
                
                pair = tuple(sorted([entity1, entity2]))
                if pair not in entity_cooccurrences:
                    entity_cooccurrences[pair] = 0
                entity_cooccurrences[pair] += 1
        
        # Add edges for related entities
        for (entity1, entity2), count in entity_cooccurrences.items():
            if count < 2:  # Require at least 2 co-occurrences
                continue
            
            entity1_type = self.graph.nodes[entity1]['type']
            entity2_type = self.graph.nodes[entity2]['type']
            
            # Create appropriate edge type based on entity types
            if entity1_type == 'topic' and entity2_type == 'topic':
                edge_type = 'related_topic'
            elif entity1_type == 'person' and entity2_type == 'organization':
                edge_type = 'affiliated_with'
            elif entity2_type == 'person' and entity1_type == 'organization':
                edge_type = 'affiliated_with'
                # Swap entities so person is source
                entity1, entity2 = entity2, entity1
            elif (entity1_type == 'topic' and entity2_type == 'legislation') or \
                 (entity2_type == 'topic' and entity1_type == 'legislation'):
                edge_type = 'related_to'
            else:
                edge_type = 'related_to'
            
            # Add bidirectional edges for most relationships
            self.add_edge(
                source_id=entity1,
                target_id=entity2,
                edge_type=edge_type,
                weight=count,
                attributes={
                    'label': edge_type.replace('_', ' ').title(),
                    'co_occurrences': count
                }
            )
            
            # Add reverse edge except for affiliated_with
            if edge_type != 'affiliated_with':
                self.add_edge(
                    source_id=entity2,
                    target_id=entity1,
                    edge_type=edge_type,
                    weight=count,
                    attributes={
                        'label': edge_type.replace('_', ' ').title(),
                        'co_occurrences': count
                    }
                )
    
    def _detect_communities(self) -> None:
        """Detect communities in the graph using the Louvain algorithm."""
        if self.graph.number_of_nodes() == 0:
            logger.warning("Cannot detect communities in empty graph")
            return
        
        # Use undirected version of graph for community detection
        undirected_graph = self.graph.to_undirected()
        
        # Set edge weights for community detection
        for u, v, d in undirected_graph.edges(data=True):
            if 'weight' not in d:
                undirected_graph[u][v]['weight'] = 1.0
        
        try:
            # Detect communities
            partition = community_louvain.best_partition(undirected_graph)
            
            # Store communities
            self.communities = partition
            
            # Group nodes by community
            community_to_nodes = {}
            for node, community_id in partition.items():
                if community_id not in community_to_nodes:
                    community_to_nodes[community_id] = []
                community_to_nodes[community_id].append(node)
            
            self.community_to_nodes = community_to_nodes
            
            # Add community attribute to nodes
            for node, community_id in partition.items():
                if node in self.graph.nodes:
                    self.graph.nodes[node]['community'] = community_id
            
            logger.info(f"Detected {len(community_to_nodes)} communities")
        
        except Exception as e:
            logger.error(f"Error detecting communities: {str(e)}")
    
    def get_nodes_by_type(self, node_type: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve.
            
        Returns:
            List of (node_id, attributes) tuples.
        """
        return [(node, attrs) for node, attrs in self.graph.nodes(data=True) 
                if attrs.get('type') == node_type]
    
    def get_node_connections(
        self, 
        node_id: str,
        max_hops: Optional[int] = None,
        relation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get connections for a specific node.
        
        Args:
            node_id: ID of the node to get connections for.
            max_hops: Maximum number of hops away from the node to traverse.
                      If None, uses the configured value.
            relation_types: Types of relations to include. If None, includes all.
            
        Returns:
            Dictionary with 'nodes' and 'edges' lists.
        """
        if node_id not in self.graph.nodes:
            logger.warning(f"Node {node_id} not found in graph")
            return {'nodes': [], 'edges': []}
        
        # Use configured max_hops if not provided
        if max_hops is None:
            max_hops = self.config.max_hops
        
        # BFS to find connected nodes
        connected_nodes = set([node_id])
        connected_edges = set()
        current_nodes = set([node_id])
        
        for _ in range(max_hops):
            next_nodes = set()
            
            for current in current_nodes:
                # Outgoing edges
                for u, v, edge_data in self.graph.out_edges(current, data=True):
                    edge_type = edge_data.get('type')
                    
                    # Skip if relation type is filtered
                    if relation_types is not None and edge_type not in relation_types:
                        continue
                    
                    # Add edge and target node
                    edge_id = f"{u}->{v}"
                    connected_edges.add(edge_id)
                    connected_nodes.add(v)
                    next_nodes.add(v)
                
                # Incoming edges
                for u, v, edge_data in self.graph.in_edges(current, data=True):
                    edge_type = edge_data.get('type')
                    
                    # Skip if relation type is filtered
                    if relation_types is not None and edge_type not in relation_types:
                        continue
                    
                    # Add edge and source node
                    edge_id = f"{u}->{v}"
                    connected_edges.add(edge_id)
                    connected_nodes.add(u)
                    next_nodes.add(u)
            
            # Update current nodes for next iteration
            current_nodes = next_nodes
            
            # Break if we've reached the maximum number of nodes
            if len(connected_nodes) >= self.config.max_nodes:
                break
        
        # Create nodes list with attributes
        nodes = []
        for node in connected_nodes:
            node_attrs = self.graph.nodes[node]
            nodes.append({
                'id': node,
                **node_attrs
            })
        
        # Create edges list with attributes
        edges = []
        for edge_id in connected_edges:
            u, v = edge_id.split('->')
            edge_attrs = self.graph.get_edge_data(u, v)
            edges.append({
                'source': u,
                'target': v,
                **edge_attrs
            })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def get_community_info(self, community_id: Optional[int] = None) -> Dict[str, Any]:
        """Get information about communities in the graph.
        
        Args:
            community_id: Specific community ID to get info for. If None, returns info for all.
            
        Returns:
            Dictionary with community information.
        """
        if not self.communities:
            logger.warning("No communities detected in graph")
            return {}
        
        if community_id is not None:
            if community_id not in self.community_to_nodes:
                logger.warning(f"Community {community_id} not found")
                return {}
            
            # Get nodes in this community
            nodes = self.community_to_nodes[community_id]
            
            # Count node types
            node_types = {}
            for node in nodes:
                node_type = self.graph.nodes[node].get('type')
                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += 1
            
            # Get key entities
            key_entities = self._get_key_entities_for_community(community_id)
            
            return {
                'community_id': community_id,
                'num_nodes': len(nodes),
                'node_types': node_types,
                'key_entities': key_entities
            }
        
        # Return info for all communities
        community_info = {}
        for comm_id in self.community_to_nodes:
            community_info[comm_id] = self.get_community_info(comm_id)
        
        return community_info
    
    def _get_key_entities_for_community(self, community_id: int) -> Dict[str, List[str]]:
        """Get key entities for a specific community.
        
        Args:
            community_id: Community ID to get key entities for.
            
        Returns:
            Dictionary mapping entity types to lists of entity labels.
        """
        if community_id not in self.community_to_nodes:
            return {}
        
        nodes = self.community_to_nodes[community_id]
        
        # Group entities by type
        entities_by_type = {}
        for node in nodes:
            node_attrs = self.graph.nodes[node]
            node_type = node_attrs.get('type')
            
            # Skip statements and sessions
            if node_type in ['statement', 'session']:
                continue
            
            if node_type not in entities_by_type:
                entities_by_type[node_type] = []
            
            entities_by_type[node_type].append({
                'id': node,
                'label': node_attrs.get('label', node),
                'degree': self.graph.degree(node)
            })
        
        # Sort entities by degree (importance)
        key_entities = {}
        for entity_type, entities in entities_by_type.items():
            # Sort by degree (descending)
            sorted_entities = sorted(entities, key=lambda x: x['degree'], reverse=True)
            
            # Take top 5 entities
            key_entities[entity_type] = [e['label'] for e in sorted_entities[:5]]
        
        return key_entities
    
    def search_by_keyword(self, keyword: str, node_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for nodes containing a keyword.
        
        Args:
            keyword: Keyword to search for.
            node_types: Types of nodes to search within. If None, searches all.
            
        Returns:
            List of matching nodes with attributes.
        """
        if not keyword:
            return []
        
        keyword = keyword.lower()
        results = []
        
        for node, attrs in self.graph.nodes(data=True):
            # Filter by node type if specified
            if node_types and attrs.get('type') not in node_types:
                continue
            
            # Check label
            label = str(attrs.get('label', '')).lower()
            if keyword in label:
                results.append({
                    'id': node,
                    **attrs
                })
                continue
            
            # Check content for statements
            if attrs.get('type') == 'statement' and 'content' in attrs:
                content = str(attrs['content']).lower()
                if keyword in content:
                    results.append({
                        'id': node,
                        **attrs
                    })
        
        return results
    
    def save_graph(self, filename: Optional[str] = None) -> bool:
        """Save the graph to a file.
        
        Args:
            filename: Name of the file to save to. If None, uses a default name.
            
        Returns:
            True if saved successfully, False otherwise.
        """
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"parliament_graph_{timestamp}.json"
        
        filepath = self.cache_dir / filename
        
        try:
            # Convert graph to JSON-serializable format
            data = nx.node_link_data(self.graph)
            
            # Add communities
            data['communities'] = self.communities
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Saved graph to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            return False
    
    def load_graph(self, filename: str) -> bool:
        """Load the graph from a file.
        
        Args:
            filename: Name of the file to load from.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        filepath = self.cache_dir / filename
        
        if not os.path.exists(filepath):
            logger.error(f"Graph file {filepath} not found")
            return False
        
        try:
            # Load from file
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract communities if present
            if 'communities' in data:
                self.communities = data['communities']
                del data['communities']
            
            # Convert JSON to graph
            self.graph = nx.node_link_graph(data)
            
            # Rebuild community_to_nodes mapping
            self.community_to_nodes = {}
            for node, community_id in self.communities.items():
                if community_id not in self.community_to_nodes:
                    self.community_to_nodes[community_id] = []
                self.community_to_nodes[community_id].append(node)
            
            logger.info(f"Loaded graph from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            return False
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph.
        
        Returns:
            Dictionary with graph statistics.
        """
        if not self.graph:
            return {}
        
        # Count nodes by type
        node_counts = {}
        for _, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('type')
            if node_type not in node_counts:
                node_counts[node_type] = 0
            node_counts[node_type] += 1
        
        # Count edges by type
        edge_counts = {}
        for _, _, attrs in self.graph.edges(data=True):
            edge_type = attrs.get('type')
            if edge_type not in edge_counts:
                edge_counts[edge_type] = 0
            edge_counts[edge_type] += 1
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'num_communities': len(self.community_to_nodes) if self.community_to_nodes else 0,
            'density': nx.density(self.graph)
        }

# Usage example:
# from src.models.graph import KnowledgeGraph
# from src.data.loader import ParliamentaryDataLoader
# from src.models.ner import EntityExtractor
# 
# # Load parliamentary data
# loader = ParliamentaryDataLoader()
# loader.load_data()
# 
# # Get a session's data
# session_data = loader.get_session_data("2024-09-10")
# 
# # Extract entities
# extractor = EntityExtractor()
# _, entity_map = extractor.extract_entities_from_dataframe(session_data)
# 
# # Build knowledge graph
# kg = KnowledgeGraph()
# kg.build_from_parliamentary_data(session_data, entity_map)
# 
# # Get graph statistics
# stats = kg.get_graph_statistics()
# print(stats)
# 
# # Save graph
# kg.save_graph() 