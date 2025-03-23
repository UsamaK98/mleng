"""
Relationship mapper module for analyzing speaker interactions
"""
from typing import Dict, Any, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import json
import os

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class RelationshipMapper:
    """
    Mapper for relationships between parliamentary speakers
    """
    def __init__(self, data_loader=None):
        """
        Initialize relationship mapper
        
        Args:
            data_loader: Data loader with minutes and speaker data
        """
        self.data_loader = data_loader
        self.minutes_df = None
        self.speakers_df = None
        self.graph = None
        
        # Check required packages
        if not NETWORKX_AVAILABLE:
            print("Warning: networkx package not available. Graph analysis will be limited.")
        
        # Load data if data_loader is provided
        if self.data_loader:
            self.load_data()
    
    def load_data(self):
        """
        Load data from the data loader
        """
        if not self.data_loader:
            raise ValueError("No data loader provided")
        
        self.minutes_df, self.speakers_df = self.data_loader.load_data()
        print(f"Loaded {len(self.minutes_df)} minutes entries and {len(self.speakers_df)} speakers")
        
        # Build the interaction graph
        self._build_interaction_graph()
    
    def _build_interaction_graph(self):
        """
        Build a directed graph of speaker interactions
        """
        if self.minutes_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if not NETWORKX_AVAILABLE:
            print("Warning: networkx package not available. Using alternative representation.")
            self.graph = self._build_interaction_dict()
            return
        
        print("Building speaker interaction graph...")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all speakers as nodes
        speakers = self.minutes_df['Speaker'].unique()
        for speaker in speakers:
            G.add_node(speaker)
            
            # Add role information if available
            if self.speakers_df is not None:
                speaker_info = self.speakers_df[self.speakers_df['Speaker'] == speaker]
                if not speaker_info.empty:
                    role = speaker_info.iloc[0].get('Role/Organization', '')
                    G.nodes[speaker]['role'] = role
        
        # Add edges based on sequential contributions
        for date in self.minutes_df['Date'].unique():
            session_df = self.minutes_df[self.minutes_df['Date'] == date].sort_values('Timestamp')
            speakers = session_df['Speaker'].tolist()
            
            for i in range(len(speakers) - 1):
                speaker1 = speakers[i]
                speaker2 = speakers[i+1]
                
                if speaker1 != speaker2:  # Avoid self-loops
                    if G.has_edge(speaker1, speaker2):
                        G[speaker1][speaker2]['weight'] += 1
                    else:
                        G.add_edge(speaker1, speaker2, weight=1)
        
        # Store the graph
        self.graph = G
        print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    def _build_interaction_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a dictionary representation of speaker interactions
        
        Returns:
            Dictionary with interaction data
        """
        interactions = defaultdict(lambda: defaultdict(int))
        
        # Add edges based on sequential contributions
        for date in self.minutes_df['Date'].unique():
            session_df = self.minutes_df[self.minutes_df['Date'] == date].sort_values('Timestamp')
            speakers = session_df['Speaker'].tolist()
            
            for i in range(len(speakers) - 1):
                speaker1 = speakers[i]
                speaker2 = speakers[i+1]
                
                if speaker1 != speaker2:  # Avoid self-loops
                    interactions[speaker1][speaker2] += 1
        
        return dict(interactions)
    
    def get_interaction_network(self, min_weight: int = 2) -> Dict[str, Any]:
        """
        Get the speaker interaction network
        
        Args:
            min_weight: Minimum interaction weight to include
            
        Returns:
            Dictionary with network data suitable for visualization
        """
        if self.graph is None:
            if not NETWORKX_AVAILABLE:
                return self._get_interaction_dict_network(min_weight)
            else:
                raise ValueError("Graph not built. Call load_data() first.")
        
        # Filter edges by weight
        filtered_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True) 
                         if d['weight'] >= min_weight]
        
        # Create filtered graph
        G_filtered = nx.DiGraph()
        G_filtered.add_nodes_from(self.graph.nodes(data=True))
        for u, v, d in filtered_edges:
            G_filtered.add_edge(u, v, **d)
        
        # Calculate node metrics
        betweenness = nx.betweenness_centrality(G_filtered, weight='weight')
        in_degree = dict(G_filtered.in_degree(weight='weight'))
        out_degree = dict(G_filtered.out_degree(weight='weight'))
        
        # Prepare nodes data
        nodes = []
        for node in G_filtered.nodes():
            role = G_filtered.nodes[node].get('role', 'Unknown')
            nodes.append({
                "id": node,
                "label": node,
                "role": role,
                "betweenness": betweenness.get(node, 0),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "total_degree": in_degree.get(node, 0) + out_degree.get(node, 0)
            })
        
        # Prepare edges data
        edges = []
        for u, v, d in G_filtered.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": d['weight']
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": G_filtered.number_of_nodes(),
                "edge_count": G_filtered.number_of_edges(),
                "avg_degree": sum(dict(G_filtered.degree()).values()) / G_filtered.number_of_nodes() if G_filtered.number_of_nodes() > 0 else 0,
                "density": nx.density(G_filtered),
                "reciprocity": nx.reciprocity(G_filtered)
            }
        }
    
    def _get_interaction_dict_network(self, min_weight: int = 2) -> Dict[str, Any]:
        """
        Get network data from the interaction dictionary
        
        Args:
            min_weight: Minimum interaction weight to include
            
        Returns:
            Dictionary with network data suitable for visualization
        """
        if isinstance(self.graph, dict):
            interactions = self.graph
        else:
            raise ValueError("Graph not built as dictionary")
        
        # Collect nodes and edges
        edges = []
        node_set = set()
        
        for source, targets in interactions.items():
            node_set.add(source)
            for target, weight in targets.items():
                if weight >= min_weight:
                    node_set.add(target)
                    edges.append({
                        "source": source,
                        "target": target,
                        "weight": weight
                    })
        
        # Prepare nodes data
        nodes = []
        for node in node_set:
            # Get role information if available
            role = "Unknown"
            if self.speakers_df is not None:
                speaker_info = self.speakers_df[self.speakers_df['Speaker'] == node]
                if not speaker_info.empty:
                    role = speaker_info.iloc[0].get('Role/Organization', 'Unknown')
            
            nodes.append({
                "id": node,
                "label": node,
                "role": role
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }
    
    def get_key_influencers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the key influencers in the network based on centrality metrics
        
        Args:
            limit: Maximum number of influencers to return
            
        Returns:
            List of key influencers with metrics
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            raise ValueError("NetworkX package required for key influencer analysis")
        
        # Calculate centrality metrics
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        in_degree = dict(self.graph.in_degree(weight='weight'))
        out_degree = dict(self.graph.out_degree(weight='weight'))
        
        # Calculate overall influence score (weighted combination)
        scores = {}
        for node in self.graph.nodes():
            scores[node] = (
                0.4 * betweenness.get(node, 0) + 
                0.3 * (in_degree.get(node, 0) / max(in_degree.values(), default=1)) + 
                0.3 * (out_degree.get(node, 0) / max(out_degree.values(), default=1))
            )
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top influencers
        influencers = []
        for node, score in sorted_scores[:limit]:
            role = self.graph.nodes[node].get('role', 'Unknown')
            influencers.append({
                "name": node,
                "role": role,
                "influence_score": score,
                "betweenness": betweenness.get(node, 0),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "total_interactions": in_degree.get(node, 0) + out_degree.get(node, 0)
            })
        
        return influencers
    
    def get_community_structure(self) -> Dict[str, Any]:
        """
        Get the community structure of the speaker network
        
        Returns:
            Dictionary with community data
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            raise ValueError("NetworkX package required for community structure analysis")

        try:
            # Convert to undirected graph for community detection
            G_undirected = self.graph.to_undirected()
            
            # Detect communities using Louvain method
            try:
                from community import best_partition
                partition = best_partition(G_undirected)
            except ImportError:
                print("Python-louvain package not available, using connected components")
                communities = list(nx.connected_components(G_undirected))
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
            
            # Organize nodes by community
            community_data = defaultdict(list)
            for node, community_id in partition.items():
                role = self.graph.nodes[node].get('role', 'Unknown')
                community_data[community_id].append({
                    "name": node,
                    "role": role
                })
            
            # Calculate community metrics
            community_metrics = {}
            for comm_id, members in community_data.items():
                member_names = [m["name"] for m in members]
                subgraph = self.graph.subgraph(member_names)
                
                # Calculate internal and external connections
                internal_edges = subgraph.number_of_edges()
                external_edges = sum(1 for u, v in self.graph.edges() 
                                    if (u in member_names and v not in member_names) or
                                      (u not in member_names and v in member_names))
                
                community_metrics[comm_id] = {
                    "size": len(members),
                    "internal_connections": internal_edges,
                    "external_connections": external_edges,
                    "cohesion": internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
                }
            
            return {
                "communities": dict(community_data),
                "metrics": community_metrics,
                "total_communities": len(community_data)
            }
            
        except Exception as e:
            print(f"Error computing community structure: {e}")
            return {"error": str(e)}
    
    def save_graph(self, output_file: str = "speaker_network.graphml"):
        """
        Save the interaction graph to a file
        
        Args:
            output_file: Path to save the graph file
        """
        if not NETWORKX_AVAILABLE or self.graph is None:
            raise ValueError("NetworkX package required for saving graph")
        
        try:
            nx.write_graphml(self.graph, output_file)
            print(f"Graph saved to {output_file}")
        except Exception as e:
            print(f"Error saving graph: {e}")
    
    def generate_visualization_data(self, min_weight: int = 2, output_file: str = "network_data.json"):
        """
        Generate data for visualization in D3.js or other tools
        
        Args:
            min_weight: Minimum interaction weight to include
            output_file: Path to save the JSON data file
        """
        # Get network data
        network_data = self.get_interaction_network(min_weight=min_weight)
        
        # Save to JSON file
        try:
            with open(output_file, 'w') as f:
                json.dump(network_data, f, indent=2)
            print(f"Visualization data saved to {output_file}")
        except Exception as e:
            print(f"Error saving visualization data: {e}")
            
        return network_data 