"""
Knowledge graph module for Parliamentary Meeting Analyzer.
Builds and manages the graph representation of parliamentary data.
"""

import os
import json
import pickle
import networkx as nx
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from src.utils.logger import log
from src.utils.config import config_manager
from src.data.loader import data_loader
from src.models.gliner_interface import gliner_interface

class KnowledgeGraph:
    """
    Knowledge graph for parliamentary data.
    Builds and manages the graph representation.
    """
    
    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = nx.MultiDiGraph()
        self.relationship_threshold = config_manager.get("graphrag.relationship_threshold", 0.6)
        self.community_detection_algorithm = config_manager.get(
            "graphrag.community_detection_algorithm", "louvain"
        )
        
        # Node types
        self.NODE_SPEAKER = "SPEAKER"
        self.NODE_SESSION = "SESSION"
        self.NODE_TOPIC = "TOPIC"
        self.NODE_ENTITY = "ENTITY"
        self.NODE_CONTENT = "CONTENT"
        
        # Edge types
        self.EDGE_SPEAKS_IN = "SPEAKS_IN"
        self.EDGE_MENTIONS = "MENTIONS"
        self.EDGE_TALKS_ABOUT = "TALKS_ABOUT"
        self.EDGE_INTERACTS_WITH = "INTERACTS_WITH"
        self.EDGE_BELONGS_TO = "BELONGS_TO"
        self.EDGE_RELATES_TO = "RELATES_TO"
        
        # Community data
        self.communities = {}
        self.community_topics = {}
        
        log.info("Knowledge graph initialized")
    
    def build_graph(self, data: Optional[pd.DataFrame] = None) -> nx.MultiDiGraph:
        """
        Build the knowledge graph from parliamentary data.
        
        Args:
            data (Optional[pd.DataFrame]): Parliamentary data.
                If None, loads data using the data_loader.
                
        Returns:
            nx.MultiDiGraph: The built graph.
        """
        # Clear existing graph
        self.graph = nx.MultiDiGraph()
        
        # Load data if not provided
        if data is None:
            data = data_loader.load_data()
        
        if data.empty:
            log.warning("No data available to build graph")
            return self.graph
        
        log.info("Building knowledge graph from parliamentary data")
        
        # Add session nodes
        sessions = data['Date'].unique()
        for session in sessions:
            self.add_node(session, self.NODE_SESSION, {"date": session})
            log.debug(f"Added session node: {session}")
        
        # Add speaker nodes
        speakers = data['Speaker'].unique()
        speaker_roles = {}
        for speaker in speakers:
            speaker_df = data[data['Speaker'] == speaker]
            roles = speaker_df['Role'].unique()
            role = roles[0] if roles[0] and roles[0] != "" else "Unknown"
            speaker_roles[speaker] = role
            
            self.add_node(speaker, self.NODE_SPEAKER, {"name": speaker, "role": role})
            log.debug(f"Added speaker node: {speaker} ({role})")
        
        # Process each session
        for session in sessions:
            session_data = data[data['Date'] == session]
            session_speakers = session_data['Speaker'].unique()
            
            # Add speaker-session edges
            for speaker in session_speakers:
                self.add_edge(speaker, session, self.EDGE_SPEAKS_IN, {"weight": 1.0})
            
            # Extract entities from session content
            session_content = " ".join(session_data['Content'].tolist())
            entities = gliner_interface.extract_entities(session_content)
            
            # Add entity nodes and relationships
            for entity in entities:
                entity_text = entity.get("text", "")
                entity_type = entity.get("label", "unknown")
                
                if not entity_text:
                    continue
                
                # Create unique ID for entity to avoid duplicates
                entity_id = f"{entity_type}:{entity_text}"
                
                # Add entity node if it doesn't exist
                if not self.node_exists(entity_id):
                    self.add_node(entity_id, self.NODE_ENTITY, {
                        "text": entity_text,
                        "type": entity_type
                    })
                
                # Add entity-session relationships
                self.add_edge(entity_id, session, self.EDGE_BELONGS_TO, {"weight": 1.0})
                
                # Find which speakers mentioned this entity
                for speaker in session_speakers:
                    speaker_content = " ".join(session_data[session_data['Speaker'] == speaker]['Content'].tolist())
                    if entity_text.lower() in speaker_content.lower():
                        self.add_edge(speaker, entity_id, self.EDGE_MENTIONS, {"weight": 1.0})
            
            # Add speaker interactions
            speaker_sequence = session_data['Speaker'].tolist()
            for i in range(1, len(speaker_sequence)):
                prev_speaker = speaker_sequence[i-1]
                current_speaker = speaker_sequence[i]
                
                if prev_speaker != current_speaker:
                    # Check if edge already exists
                    if self.edge_exists(prev_speaker, current_speaker, self.EDGE_INTERACTS_WITH):
                        # Increment weight of existing edge
                        for _, _, edge_data in self.graph.edges(data=True):
                            if (edge_data.get('edge_type') == self.EDGE_INTERACTS_WITH and 
                                edge_data.get('source') == prev_speaker and 
                                edge_data.get('target') == current_speaker):
                                edge_data['weight'] += 1.0
                    else:
                        # Add new edge
                        self.add_edge(prev_speaker, current_speaker, self.EDGE_INTERACTS_WITH, {"weight": 1.0})
        
        # Detect communities
        self._detect_communities()
        
        log.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and "
                 f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id (str): Unique identifier for the node.
            node_type (str): Type of the node.
            properties (Dict[str, Any]): Node properties.
        """
        # Make a copy of properties to avoid modifying the original dict
        props = properties.copy()
        
        # Set node_type separately to avoid conflicts with any 'type' in properties
        if node_id not in self.graph:
            self.graph.add_node(node_id)
            # Update node attributes after adding
            nx.set_node_attributes(self.graph, {node_id: {"node_type": node_type, **props}})
    
    def add_edge(self, source: str, target: str, edge_type: str, properties: Dict[str, Any]) -> None:
        """
        Add an edge to the graph.
        
        Args:
            source (str): Source node ID.
            target (str): Target node ID.
            edge_type (str): Type of the edge.
            properties (Dict[str, Any]): Edge properties.
        """
        # Make a copy of properties to avoid modifying the original dict
        props = properties.copy()
        
        # Check if edge exists, different types are allowed between same nodes
        if not self.edge_exists(source, target, edge_type):
            self.graph.add_edge(source, target, edge_type=edge_type, source=source, target=target, **props)
    
    def node_exists(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.
        
        Args:
            node_id (str): Node ID to check.
            
        Returns:
            bool: True if the node exists, False otherwise.
        """
        return node_id in self.graph
    
    def edge_exists(self, source: str, target: str, edge_type: str) -> bool:
        """
        Check if an edge exists in the graph.
        
        Args:
            source (str): Source node ID.
            target (str): Target node ID.
            edge_type (str): Type of the edge.
            
        Returns:
            bool: True if the edge exists, False otherwise.
        """
        if not self.node_exists(source) or not self.node_exists(target):
            return False
        
        for _, _, edge_data in self.graph.edges(data=True):
            if (edge_data.get('edge_type') == edge_type and
                edge_data.get('source') == source and
                edge_data.get('target') == target):
                return True
        
        return False
    
    def _detect_communities(self) -> Dict[str, int]:
        """
        Detect communities in the graph.
        
        Returns:
            Dict[str, int]: Mapping of node IDs to community IDs.
        """
        if self.graph.number_of_nodes() == 0:
            log.warning("Cannot detect communities: graph is empty")
            return {}
        
        log.info("Detecting communities in the knowledge graph")
        
        # Create undirected graph for community detection
        undirected_graph = nx.Graph()
        
        # Copy nodes
        for node, node_data in self.graph.nodes(data=True):
            undirected_graph.add_node(node, **node_data)
        
        # Copy edges (combining parallel edges by summing weights)
        edge_weights = {}
        for source, target, edge_data in self.graph.edges(data=True):
            key = (source, target) if source < target else (target, source)
            weight = edge_data.get('weight', 1.0)
            
            if key in edge_weights:
                edge_weights[key] += weight
            else:
                edge_weights[key] = weight
        
        # Add weighted edges to undirected graph
        for (source, target), weight in edge_weights.items():
            undirected_graph.add_edge(source, target, weight=weight)
        
        # Detect communities using the specified algorithm
        if self.community_detection_algorithm == "louvain":
            try:
                from community import best_partition
                self.communities = best_partition(undirected_graph)
            except ImportError:
                log.warning("Louvain algorithm not available. Using label propagation instead.")
                self.communities = {node: i for i, community in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(undirected_graph)) for node in community}
        else:
            # Default to label propagation
            self.communities = {node: i for i, community in enumerate(nx.algorithms.community.label_propagation.label_propagation_communities(undirected_graph)) for node in community}
        
        # Assign community information to nodes
        for node, community_id in self.communities.items():
            if node in self.graph:
                nx.set_node_attributes(self.graph, {node: {"community": community_id}})
        
        # Extract topics for each community
        self._extract_community_topics()
        
        log.info(f"Detected {len(set(self.communities.values()))} communities")
        return self.communities
    
    def _extract_community_topics(self) -> Dict[int, List[str]]:
        """
        Extract representative topics for each community.
        
        Returns:
            Dict[int, List[str]]: Mapping of community IDs to topic lists.
        """
        self.community_topics = {}
        
        # Group nodes by community
        community_nodes = {}
        for node, community_id in self.communities.items():
            if community_id not in community_nodes:
                community_nodes[community_id] = []
            community_nodes[community_id].append(node)
        
        # Extract topics for each community
        for community_id, nodes in community_nodes.items():
            # Get entity nodes in this community
            entity_nodes = []
            for node in nodes:
                if node in self.graph and self.graph.nodes[node].get('node_type') == self.NODE_ENTITY:
                    entity_nodes.append(node)
            
            # Extract top entities as topics
            topics = []
            for node in entity_nodes:
                entity_type = self.graph.nodes[node].get('node_type')
                entity_text = self.graph.nodes[node].get('text', '')
                
                if entity_type and entity_text:
                    topics.append(entity_text)
            
            self.community_topics[community_id] = topics[:10]  # Top 10 topics
        
        return self.community_topics
    
    def get_node_community(self, node_id: str) -> Optional[int]:
        """
        Get the community ID for a node.
        
        Args:
            node_id (str): Node ID.
            
        Returns:
            Optional[int]: Community ID or None if not found.
        """
        return self.communities.get(node_id)
    
    def get_community_nodes(self, community_id: int) -> List[str]:
        """
        Get all nodes in a community.
        
        Args:
            community_id (int): Community ID.
            
        Returns:
            List[str]: List of node IDs in the community.
        """
        return [node for node, cid in self.communities.items() if cid == community_id]
    
    def get_community_topics(self, community_id: int) -> List[str]:
        """
        Get representative topics for a community.
        
        Args:
            community_id (int): Community ID.
            
        Returns:
            List[str]: List of topics.
        """
        return self.community_topics.get(community_id, [])
    
    def get_node_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """
        Get neighboring nodes.
        
        Args:
            node_id (str): Node ID.
            edge_type (Optional[str]): Filter by edge type.
            
        Returns:
            List[str]: List of neighboring node IDs.
        """
        if not self.node_exists(node_id):
            return []
        
        neighbors = []
        
        for source, target, edge_data in self.graph.edges(data=True):
            edge_type_value = edge_data.get('edge_type')
            
            if edge_type and edge_type_value != edge_type:
                continue
                
            if source == node_id:
                neighbors.append(target)
            elif target == node_id:
                neighbors.append(source)
        
        return neighbors
    
    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type (str): Type of nodes to return.
            
        Returns:
            List[str]: List of node IDs.
        """
        return [node for node, data in self.graph.nodes(data=True) if data.get('node_type') == node_type]
    
    def save_graph(self, file_path: Optional[str] = None) -> bool:
        """
        Save the graph to a file.
        
        Args:
            file_path (Optional[str]): File path to save to.
                If None, uses the path specified in the configuration.
                
        Returns:
            bool: True if successful, False otherwise.
        """
        if file_path is None:
            file_path = config_manager.get("paths.output.graph", "data/graph/knowledge_graph.pkl")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            # Save the graph
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'communities': self.communities,
                    'community_topics': self.community_topics
                }, f)
            
            log.info(f"Graph saved to {file_path}")
            return True
        except Exception as e:
            log.error(f"Error saving graph: {e}")
            return False
    
    def load_graph(self, file_path: Optional[str] = None) -> bool:
        """
        Load the graph from a file.
        
        Args:
            file_path (Optional[str]): File path to load from.
                If None, uses the path specified in the configuration.
                
        Returns:
            bool: True if successful, False otherwise.
        """
        if file_path is None:
            file_path = config_manager.get("paths.output.graph", "data/graph/knowledge_graph.pkl")
        
        if not os.path.exists(file_path):
            log.warning(f"Graph file not found: {file_path}")
            return False
        
        try:
            # Load the graph
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.graph = data.get('graph', nx.MultiDiGraph())
                self.communities = data.get('communities', {})
                self.community_topics = data.get('community_topics', {})
            
            log.info(f"Graph loaded from {file_path}")
            return True
        except Exception as e:
            log.error(f"Error loading graph: {e}")
            return False


# Create a singleton instance
knowledge_graph = KnowledgeGraph() 