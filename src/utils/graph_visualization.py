"""
Graph Visualization Utilities for Parliamentary Meeting Analyzer.

This module provides functions to visualize and export knowledge graphs
for visualization in various formats.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

from src.utils.logging import logger
from src.utils.config import config_manager
from src.models.graph import KnowledgeGraph

def export_graph_for_d3(
    kg: KnowledgeGraph,
    output_path: Optional[str] = None,
    max_nodes: int = 100,
    include_communities: bool = True
) -> Dict[str, Any]:
    """Export graph in a format suitable for D3.js visualization.
    
    Args:
        kg: KnowledgeGraph instance.
        output_path: Path to save the JSON file. If None, doesn't save to file.
        max_nodes: Maximum number of nodes to include.
        include_communities: Whether to include community information.
        
    Returns:
        Dictionary containing graph data in D3.js format.
    """
    if not output_path:
        output_path = os.path.join(config_manager.config.output_dir, "graph_data_d3.json")
    
    # Get graph statistics to guide filtering
    stats = kg.get_graph_statistics()
    logger.info(f"Exporting graph with {stats.get('num_nodes', 0)} nodes and {stats.get('num_edges', 0)} edges")
    
    # Limit nodes if too many
    if stats.get('num_nodes', 0) > max_nodes:
        logger.info(f"Limiting export to {max_nodes} nodes")
    
    # Create a subgraph if needed
    if stats.get('num_nodes', 0) > max_nodes:
        # Get important node types
        important_nodes = []
        
        # Include all speakers (person nodes)
        speakers = kg.get_nodes_by_type("person")
        important_nodes.extend([node_id for node_id, _ in speakers])
        
        # Include all topics
        topics = kg.get_nodes_by_type("topic")
        important_nodes.extend([node_id for node_id, _ in topics])
        
        # Include all organizations
        orgs = kg.get_nodes_by_type("organization")
        important_nodes.extend([node_id for node_id, _ in orgs])
        
        # If we still have room, add statements
        if len(important_nodes) < max_nodes:
            # Add statements until we reach max_nodes
            statements = kg.get_nodes_by_type("statement")
            remaining = max_nodes - len(important_nodes)
            
            # Prefer statements with more connections
            statement_degrees = [
                (node_id, kg.graph.degree(node_id)) 
                for node_id, _ in statements
            ]
            statement_degrees.sort(key=lambda x: x[1], reverse=True)
            
            for node_id, _ in statement_degrees[:remaining]:
                important_nodes.append(node_id)
        
        # Create subgraph with only the selected nodes
        subgraph = kg.graph.subgraph(important_nodes)
    else:
        subgraph = kg.graph
    
    # Convert to D3.js format
    nodes = []
    node_ids = {}  # Map node IDs to indices
    
    # Process nodes
    for i, (node_id, attrs) in enumerate(subgraph.nodes(data=True)):
        node_data = {
            "id": node_id,
            "name": attrs.get("label", node_id),
            "type": attrs.get("type", "unknown"),
            "degree": subgraph.degree(node_id)
        }
        
        # Add community information if available
        if include_communities and "community" in attrs:
            node_data["community"] = attrs["community"]
        
        # Include relevant attributes based on node type
        if attrs.get("type") == "statement":
            if "content" in attrs:
                # Truncate long content
                content = attrs["content"]
                node_data["content"] = content[:200] + "..." if len(content) > 200 else content
            
            if "date" in attrs:
                node_data["date"] = attrs["date"]
        
        nodes.append(node_data)
        node_ids[node_id] = i
    
    # Process edges
    links = []
    for source, target, attrs in subgraph.edges(data=True):
        if source in node_ids and target in node_ids:
            link_data = {
                "source": node_ids[source],
                "target": node_ids[target],
                "type": attrs.get("type", "unknown"),
                "weight": attrs.get("weight", 1.0)
            }
            
            if "label" in attrs:
                link_data["label"] = attrs["label"]
            
            links.append(link_data)
    
    # Create result
    result = {
        "nodes": nodes,
        "links": links
    }
    
    # Save to file if output_path is provided
    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Graph data exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting graph data: {str(e)}")
    
    return result

def export_graph_for_gephi(
    kg: KnowledgeGraph,
    output_path: Optional[str] = None
) -> Tuple[str, str]:
    """Export graph in GEXF format for Gephi visualization.
    
    Args:
        kg: KnowledgeGraph instance.
        output_path: Path to save the GEXF file. If None, uses default path.
        
    Returns:
        Tuple containing paths to nodes and edges CSV files.
    """
    if not output_path:
        output_dir = config_manager.config.output_dir
        gexf_path = os.path.join(output_dir, "parliament_graph.gexf")
    else:
        gexf_path = output_path
    
    try:
        # Export to GEXF format
        nx.write_gexf(kg.graph, gexf_path)
        logger.info(f"Graph exported to GEXF format at {gexf_path}")
        
        # Also export as CSV for more compatibility
        nodes_csv = os.path.splitext(gexf_path)[0] + "_nodes.csv"
        edges_csv = os.path.splitext(gexf_path)[0] + "_edges.csv"
        
        # Create nodes dataframe
        nodes_data = []
        for node, attrs in kg.graph.nodes(data=True):
            node_data = {"id": node}
            node_data.update(attrs)
            nodes_data.append(node_data)
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(nodes_csv, index=False)
        
        # Create edges dataframe
        edges_data = []
        for source, target, attrs in kg.graph.edges(data=True):
            edge_data = {"source": source, "target": target}
            edge_data.update(attrs)
            edges_data.append(edge_data)
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(edges_csv, index=False)
        
        logger.info(f"Graph nodes exported to CSV at {nodes_csv}")
        logger.info(f"Graph edges exported to CSV at {edges_csv}")
        
        return nodes_csv, edges_csv
    
    except Exception as e:
        logger.error(f"Error exporting graph for Gephi: {str(e)}")
        return "", ""

def visualize_graph_matplotlib(
    kg: KnowledgeGraph,
    output_path: Optional[str] = None,
    max_nodes: int = 50,
    show_labels: bool = True,
    node_size_factor: float = 100.0,
    edge_width_factor: float = 1.0,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """Visualize graph using matplotlib.
    
    Args:
        kg: KnowledgeGraph instance.
        output_path: Path to save the visualization. If None, displays it.
        max_nodes: Maximum number of nodes to include.
        show_labels: Whether to show node labels.
        node_size_factor: Factor to adjust node sizes.
        edge_width_factor: Factor to adjust edge widths.
        figsize: Figure size.
    """
    # Create a subgraph with the most important nodes
    if kg.graph.number_of_nodes() > max_nodes:
        # Get important nodes similar to D3 export function
        important_nodes = []
        
        # Include all speakers (person nodes)
        speakers = kg.get_nodes_by_type("person")
        important_nodes.extend([node_id for node_id, _ in speakers])
        
        # Include all topics
        topics = kg.get_nodes_by_type("topic")
        important_nodes.extend([node_id for node_id, _ in topics])
        
        # Include all organizations
        orgs = kg.get_nodes_by_type("organization")
        important_nodes.extend([node_id for node_id, _ in orgs])
        
        # If we still have room, add statements
        if len(important_nodes) < max_nodes:
            # Add statements until we reach max_nodes
            statements = kg.get_nodes_by_type("statement")
            remaining = max_nodes - len(important_nodes)
            
            # Prefer statements with more connections
            statement_degrees = [
                (node_id, kg.graph.degree(node_id)) 
                for node_id, _ in statements
            ]
            statement_degrees.sort(key=lambda x: x[1], reverse=True)
            
            for node_id, _ in statement_degrees[:remaining]:
                important_nodes.append(node_id)
        
        # Create subgraph with only the selected nodes
        g = kg.graph.subgraph(important_nodes)
    else:
        g = kg.graph
    
    # Set up colors for node types
    node_colors = {
        "person": "skyblue",
        "topic": "lightgreen",
        "organization": "orange",
        "statement": "lightgray",
        "legislation": "pink",
        "location": "yellow",
        "session": "purple"
    }
    
    # Set up colors for edge types
    edge_colors = {
        "made_statement": "blue",
        "about_topic": "green",
        "mentions_person": "red",
        "mentions_organization": "orange",
        "participated_in": "purple",
        "part_of": "gray",
        "responds_to": "black",
        "references_legislation": "pink",
        "mentions_location": "brown",
        "related_topic": "lightgreen",
        "affiliated_with": "darkblue"
    }
    
    # Extract node positions using spring layout
    logger.info("Computing graph layout...")
    pos = nx.spring_layout(g, k=0.3, iterations=50)
    
    # Set up plot
    plt.figure(figsize=figsize)
    
    # Draw nodes
    node_colors_list = []
    node_sizes = []
    
    for node in g.nodes():
        node_type = g.nodes[node].get("type", "unknown")
        color = node_colors.get(node_type, "gray")
        node_colors_list.append(color)
        
        # Size node by degree
        size = g.degree(node) * node_size_factor
        size = max(size, 300)  # Minimum size
        node_sizes.append(size)
    
    # Draw edges with colors based on type
    for edge_type in set(nx.get_edge_attributes(g, "type").values()):
        edge_list = [(u, v) for u, v, d in g.edges(data=True) if d.get("type") == edge_type]
        edge_color = edge_colors.get(edge_type, "gray")
        
        # Get edge weights
        edge_weights = []
        for u, v in edge_list:
            weight = g[u][v].get("weight", 1.0)
            edge_weights.append(weight * edge_width_factor)
        
        if edge_list:
            nx.draw_networkx_edges(
                g, pos, 
                edgelist=edge_list, 
                width=edge_weights,
                edge_color=edge_color, 
                alpha=0.7
            )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        g, pos,
        node_color=node_colors_list,
        node_size=node_sizes,
        alpha=0.8
    )
    
    # Draw labels if requested
    if show_labels:
        labels = {}
        for node in g.nodes():
            # Use label attribute if available, otherwise use truncated ID
            label = g.nodes[node].get("label", node)
            if isinstance(label, str) and len(label) > 15:
                label = label[:12] + "..."
            labels[node] = label
        
        nx.draw_networkx_labels(
            g, pos,
            labels=labels,
            font_size=8,
            font_color="black"
        )
    
    # Add a legend for node types
    legend_elements = []
    for node_type, color in node_colors.items():
        legend_elements.append(plt.Line2D(
            [0], [0], 
            marker='o', 
            color='w', 
            markerfacecolor=color, 
            markersize=10, 
            label=node_type
        ))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Graph visualization saved to {output_path}")
    else:
        plt.show()

def create_plotly_graph(
    kg: KnowledgeGraph,
    max_nodes: int = 100
) -> go.Figure:
    """Create an interactive plotly graph visualization.
    
    Args:
        kg: KnowledgeGraph instance.
        max_nodes: Maximum number of nodes to include.
        
    Returns:
        Plotly Figure object.
    """
    # Create a subgraph with important nodes as in other functions
    if kg.graph.number_of_nodes() > max_nodes:
        # Select important nodes
        important_nodes = []
        
        # Include all speakers, topics, and organizations
        for node_type in ["person", "topic", "organization"]:
            nodes = kg.get_nodes_by_type(node_type)
            important_nodes.extend([node_id for node_id, _ in nodes])
        
        # Include high-degree statements if space allows
        if len(important_nodes) < max_nodes:
            statements = kg.get_nodes_by_type("statement")
            statement_degrees = [
                (node_id, kg.graph.degree(node_id)) 
                for node_id, _ in statements
            ]
            statement_degrees.sort(key=lambda x: x[1], reverse=True)
            
            remaining = max_nodes - len(important_nodes)
            important_nodes.extend([node_id for node_id, _ in statement_degrees[:remaining]])
        
        g = kg.graph.subgraph(important_nodes)
    else:
        g = kg.graph
    
    # Node color map
    node_colors = {
        "person": "#3366CC",
        "topic": "#33CC33",
        "organization": "#FF9900",
        "statement": "#CCCCCC",
        "legislation": "#FF66CC",
        "location": "#FFFF33",
        "session": "#9966CC"
    }
    
    # Edge color map
    edge_colors = {
        "made_statement": "#3366CC",
        "about_topic": "#33CC33",
        "mentions_person": "#CC3333",
        "mentions_organization": "#FF9900",
        "participated_in": "#9966CC",
        "part_of": "#999999",
        "responds_to": "#000000",
        "related_topic": "#99FF99"
    }
    
    # Compute layout
    logger.info("Computing graph layout for visualization...")
    pos = nx.spring_layout(g, dim=3, seed=42)
    
    # Prepare node trace
    node_x, node_y, node_z = [], [], []
    node_text, node_color, node_size = [], [], []
    
    for node, attrs in g.nodes(data=True):
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        # Node color based on type
        node_type = attrs.get("type", "unknown")
        color = node_colors.get(node_type, "#CCCCCC")
        node_color.append(color)
        
        # Node size based on degree
        size = g.degree(node) * 2 + 5
        node_size.append(size)
        
        # Node text
        label = attrs.get("label", node)
        if node_type == "statement" and "content" in attrs:
            content = attrs["content"]
            if len(content) > 50:
                content = content[:47] + "..."
            text = f"{label}<br>{content}"
        else:
            text = label
        node_text.append(text)
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='#FFFFFF')
        ),
        text=node_text,
        hoverinfo='text'
    )
    
    # Prepare edge traces (group by type for coloring)
    edge_traces = []
    
    # Group edges by type
    edge_by_type = {}
    for u, v, attrs in g.edges(data=True):
        edge_type = attrs.get("type", "unknown")
        if edge_type not in edge_by_type:
            edge_by_type[edge_type] = []
        edge_by_type[edge_type].append((u, v, attrs))
    
    # Create a trace for each edge type
    for edge_type, edges in edge_by_type.items():
        edge_x, edge_y, edge_z = [], [], []
        edge_text = []
        
        for u, v, attrs in edges:
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            
            # Add line (with None as separator between edges)
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
            # Edge text
            label = attrs.get("label", edge_type)
            source_label = g.nodes[u].get("label", u)
            target_label = g.nodes[v].get("label", v)
            text = f"{source_label} {label} {target_label}"
            edge_text.append(text)
        
        color = edge_colors.get(edge_type, "#CCCCCC")
        
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color=color, width=1),
            hoverinfo='none',
            name=edge_type
        )
        
        edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=[*edge_traces, node_trace])
    
    # Update layout
    fig.update_layout(
        title='Parliamentary Knowledge Graph',
        showlegend=False,
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title='')
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def export_community_data(
    kg: KnowledgeGraph,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Export community data for analysis and visualization.
    
    Args:
        kg: KnowledgeGraph instance.
        output_path: Path to save the community data. If None, doesn't save to file.
        
    Returns:
        Dictionary containing community data.
    """
    if not output_path:
        output_path = os.path.join(config_manager.config.output_dir, "community_data.json")
    
    # Get community information
    community_info = kg.get_community_info()
    
    # Process community data for export
    community_data = []
    
    for community_id, info in community_info.items():
        # Convert community ID to string for JSON compatibility
        comm_id_str = str(community_id)
        
        # Basic community information
        comm_data = {
            "community_id": comm_id_str,
            "size": info.get("num_nodes", 0),
            "node_types": info.get("node_types", {})
        }
        
        # Key entities
        comm_data["key_entities"] = info.get("key_entities", {})
        
        # Add to list
        community_data.append(comm_data)
    
    # Create final result
    result = {
        "total_communities": len(community_data),
        "communities": community_data
    }
    
    # Save to file if path provided
    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Community data exported to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting community data: {str(e)}")
    
    return result

# Example usage:
# from src.utils.graph_visualization import export_graph_for_d3, visualize_graph_matplotlib
# from src.models.graph import KnowledgeGraph
# 
# # Initialize and build knowledge graph...
# 
# # Export for web visualization
# graph_data = export_graph_for_d3(kg, "web/static/graph_data.json")
# 
# # Create a static visualization
# visualize_graph_matplotlib(kg, "output/graph_viz.png", max_nodes=50) 