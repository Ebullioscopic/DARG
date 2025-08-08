"""
Neo4j Graph Visualization for Universal DARG
============================================

Real-time graph visualization and exploration of vector relationships
using Neo4j graph database with advanced analytics capabilities.
"""

import json
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Mock Neo4j implementation for systems without Neo4j
class MockNeo4jDriver:
    """Mock Neo4j driver for testing without actual Neo4j installation"""
    
    def __init__(self, uri: str, auth: Tuple[str, str]):
        self.uri = uri
        self.auth = auth
        self.data = {
            'nodes': {},
            'relationships': [],
            'indexes': [],
            'constraints': []
        }
        logger.info(f"Mock Neo4j driver initialized for {uri}")
    
    def session(self):
        return MockNeo4jSession(self.data)
    
    def close(self):
        logger.info("Mock Neo4j driver closed")

class MockNeo4jSession:
    """Mock Neo4j session for testing"""
    
    def __init__(self, data: Dict):
        self.data = data
    
    def run(self, query: str, **params):
        logger.debug(f"Mock Neo4j query: {query[:100]}...")
        
        # Simple simulation of common queries
        if "CREATE CONSTRAINT" in query or "CREATE INDEX" in query:
            return MockNeo4jResult([])
        elif "MERGE" in query and "DataNode" in query:
            # Add node
            node_id = params.get('id', 'unknown')
            self.data['nodes'][node_id] = params
            return MockNeo4jResult([])
        elif "SIMILAR_TO" in query:
            # Add relationship
            rel = {
                'id1': params.get('id1'),
                'id2': params.get('id2'),
                'similarity': params.get('similarity', 0.0)
            }
            self.data['relationships'].append(rel)
            return MockNeo4jResult([])
        elif "count(DISTINCT n)" in query:
            # Statistics query
            return MockNeo4jResult([{
                'node_count': len(self.data['nodes']),
                'edge_count': len(self.data['relationships']),
                'data_types': list(set(n.get('data_type', 'unknown') for n in self.data['nodes'].values()))
            }])
        else:
            return MockNeo4jResult([])
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockNeo4jResult:
    """Mock Neo4j result for testing"""
    
    def __init__(self, records: List[Dict]):
        self.records = records
    
    def single(self):
        return self.records[0] if self.records else {}
    
    def data(self):
        return self.records

# Try to import real Neo4j, fall back to mock
try:
    from neo4j import GraphDatabase
    Neo4jDriver = GraphDatabase.driver
    logger.info("Using real Neo4j driver")
except ImportError:
    Neo4jDriver = MockNeo4jDriver
    logger.info("Using mock Neo4j driver (install neo4j package for real functionality)")

class AdvancedNeo4jVisualizer:
    """Advanced Neo4j visualizer with analytics and real-time updates"""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password",
                 database: str = "neo4j"):
        
        self.uri = uri
        self.user = user
        self.database = database
        
        try:
            self.driver = Neo4jDriver(uri, auth=(user, password))
            # Test connection with a simple query
            with self.driver.session() as session:
                session.run("RETURN 1")
            self._setup_database()
            logger.info(f"Neo4j visualizer connected to {uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            logger.info("Using mock Neo4j driver instead")
            self.driver = MockNeo4jDriver(uri, (user, password))
    
    def _setup_database(self):
        """Set up Neo4j database schema"""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DataNode) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Partition) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Cluster) REQUIRE n.id IS UNIQUE"
            ]
            
            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (n:DataNode) ON (n.data_type)",
                "CREATE INDEX IF NOT EXISTS FOR (n:DataNode) ON (n.timestamp)",
                "CREATE INDEX IF NOT EXISTS FOR (n:DataNode) ON (n.local_id)",
                "CREATE INDEX IF NOT EXISTS FOR (n:Partition) ON (n.partition_id)",
                "CREATE INDEX IF NOT EXISTS FOR (r:SIMILAR_TO) ON (r.similarity)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint creation info: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    logger.debug(f"Index creation info: {e}")
    
    def add_data_node(self, 
                     node_id: str,
                     data_type: str,
                     vector_dim: int,
                     timestamp: float,
                     metadata: Dict[str, Any],
                     local_id: float = None,
                     partition_id: int = None):
        """Add a data node with enhanced properties"""
        
        with self.driver.session() as session:
            session.run("""
                MERGE (n:DataNode {id: $id})
                SET n.data_type = $data_type,
                    n.timestamp = $timestamp,
                    n.vector_dim = $vector_dim,
                    n.metadata = $metadata,
                    n.local_id = $local_id,
                    n.partition_id = $partition_id,
                    n.last_updated = $last_updated
            """, 
            id=node_id,
            data_type=data_type,
            timestamp=timestamp,
            vector_dim=vector_dim,
            metadata=json.dumps(metadata),
            local_id=local_id,
            partition_id=partition_id,
            last_updated=time.time()
            )
    
    def add_partition_node(self, partition_id: int, node_count: int, avg_local_id: float):
        """Add a partition node"""
        with self.driver.session() as session:
            session.run("""
                MERGE (p:Partition {id: $partition_id})
                SET p.partition_id = $partition_id,
                    p.node_count = $node_count,
                    p.avg_local_id = $avg_local_id,
                    p.last_updated = $last_updated
            """,
            partition_id=partition_id,
            node_count=node_count,
            avg_local_id=avg_local_id,
            last_updated=time.time()
            )
    
    def add_similarity_relationship(self, 
                                  node1_id: str, 
                                  node2_id: str, 
                                  similarity: float,
                                  relationship_type: str = "SIMILAR_TO"):
        """Add similarity relationship between nodes"""
        with self.driver.session() as session:
            session.run(f"""
                MATCH (a:DataNode {{id: $id1}}), (b:DataNode {{id: $id2}})
                MERGE (a)-[r:{relationship_type}]-(b)
                SET r.similarity = $similarity,
                    r.distance = $distance,
                    r.strength = $strength,
                    r.last_updated = $last_updated
            """, 
            id1=node1_id, 
            id2=node2_id, 
            similarity=similarity,
            distance=1.0 - similarity,
            strength=similarity ** 2,  # Quadratic strength for visualization
            last_updated=time.time()
            )
    
    def add_partition_membership(self, node_id: str, partition_id: int):
        """Add node to partition relationship"""
        with self.driver.session() as session:
            session.run("""
                MATCH (n:DataNode {id: $node_id}), (p:Partition {id: $partition_id})
                MERGE (n)-[r:BELONGS_TO]->(p)
                SET r.timestamp = $timestamp
            """,
            node_id=node_id,
            partition_id=partition_id,
            timestamp=time.time()
            )
    
    def get_graph_analytics(self) -> Dict[str, Any]:
        """Get comprehensive graph analytics"""
        with self.driver.session() as session:
            # Basic statistics
            basic_stats = session.run("""
                MATCH (n:DataNode)
                OPTIONAL MATCH (n)-[r:SIMILAR_TO]-()
                RETURN 
                    count(DISTINCT n) as total_nodes,
                    count(r)/2 as total_edges,
                    collect(DISTINCT n.data_type) as data_types,
                    avg(n.local_id) as avg_local_id,
                    count(DISTINCT n.partition_id) as total_partitions
            """).single()
            
            # Similarity distribution
            similarity_stats = session.run("""
                MATCH ()-[r:SIMILAR_TO]-()
                RETURN 
                    min(r.similarity) as min_similarity,
                    max(r.similarity) as max_similarity,
                    avg(r.similarity) as avg_similarity,
                    percentileCont(r.similarity, 0.5) as median_similarity,
                    percentileCont(r.similarity, 0.95) as p95_similarity
            """).single()
            
            # Data type distribution
            type_distribution = session.run("""
                MATCH (n:DataNode)
                RETURN n.data_type as data_type, count(n) as count
                ORDER BY count DESC
            """).data()
            
            # Partition statistics
            partition_stats = session.run("""
                MATCH (n:DataNode)
                WHERE n.partition_id IS NOT NULL
                RETURN 
                    n.partition_id as partition_id,
                    count(n) as node_count,
                    avg(n.local_id) as avg_local_id,
                    min(n.timestamp) as earliest_timestamp,
                    max(n.timestamp) as latest_timestamp
                ORDER BY node_count DESC
                LIMIT 10
            """).data()
            
            # Network metrics
            network_metrics = session.run("""
                MATCH (n:DataNode)
                OPTIONAL MATCH (n)-[r:SIMILAR_TO]-()
                RETURN 
                    n.id as node_id,
                    count(r) as degree,
                    n.data_type as data_type,
                    n.local_id as local_id
                ORDER BY degree DESC
                LIMIT 10
            """).data()
            
            return {
                'basic_stats': dict(basic_stats) if basic_stats else {},
                'similarity_stats': dict(similarity_stats) if similarity_stats else {},
                'type_distribution': type_distribution,
                'partition_stats': partition_stats,
                'top_connected_nodes': network_metrics,
                'analysis_timestamp': time.time()
            }
    
    def find_similar_clusters(self, min_similarity: float = 0.8, min_cluster_size: int = 3) -> List[Dict]:
        """Find clusters of highly similar nodes"""
        with self.driver.session() as session:
            clusters = session.run("""
                MATCH (n:DataNode)-[r:SIMILAR_TO]-(m:DataNode)
                WHERE r.similarity >= $min_similarity
                WITH n, collect(DISTINCT m.id) as similar_nodes
                WHERE size(similar_nodes) >= $min_cluster_size
                RETURN 
                    n.id as center_node,
                    n.data_type as data_type,
                    n.local_id as local_id,
                    similar_nodes,
                    size(similar_nodes) as cluster_size
                ORDER BY cluster_size DESC
                LIMIT 20
            """, 
            min_similarity=min_similarity,
            min_cluster_size=min_cluster_size
            ).data()
            
            return clusters
    
    def get_node_neighborhood(self, node_id: str, max_hops: int = 2) -> Dict[str, Any]:
        """Get neighborhood information for a specific node"""
        with self.driver.session() as session:
            neighborhood = session.run("""
                MATCH (center:DataNode {id: $node_id})
                OPTIONAL MATCH path = (center)-[r:SIMILAR_TO*1..$max_hops]-(neighbor:DataNode)
                WITH center, 
                     collect(DISTINCT neighbor) as neighbors,
                     collect(DISTINCT r) as relationships
                RETURN 
                    center.id as center_id,
                    center.data_type as center_type,
                    center.local_id as center_local_id,
                    [n IN neighbors | {
                        id: n.id, 
                        data_type: n.data_type, 
                        local_id: n.local_id
                    }] as neighbors,
                    size(neighbors) as neighbor_count
            """,
            node_id=node_id,
            max_hops=max_hops
            ).single()
            
            return dict(neighborhood) if neighborhood else {}
    
    def export_graph_data(self, format: str = "json") -> Dict[str, Any]:
        """Export graph data for external visualization"""
        with self.driver.session() as session:
            # Export nodes
            nodes = session.run("""
                MATCH (n:DataNode)
                RETURN 
                    n.id as id,
                    n.data_type as data_type,
                    n.timestamp as timestamp,
                    n.local_id as local_id,
                    n.partition_id as partition_id
            """).data()
            
            # Export relationships
            relationships = session.run("""
                MATCH (a:DataNode)-[r:SIMILAR_TO]-(b:DataNode)
                WHERE id(a) < id(b)  // Avoid duplicates
                RETURN 
                    a.id as source,
                    b.id as target,
                    r.similarity as similarity,
                    r.distance as distance,
                    r.strength as strength
            """).data()
            
            graph_data = {
                'nodes': nodes,
                'relationships': relationships,
                'metadata': {
                    'export_timestamp': time.time(),
                    'total_nodes': len(nodes),
                    'total_edges': len(relationships),
                    'format': format
                }
            }
            
            if format == "cytoscape":
                # Convert to Cytoscape.js format
                cytoscape_data = {
                    'nodes': [{'data': node} for node in nodes],
                    'edges': [{'data': rel} for rel in relationships]
                }
                return cytoscape_data
            
            return graph_data
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old data from the graph"""
        cutoff_timestamp = time.time() - (days_old * 24 * 60 * 60)
        
        with self.driver.session() as session:
            # Remove old nodes and their relationships
            result = session.run("""
                MATCH (n:DataNode)
                WHERE n.timestamp < $cutoff_timestamp
                DETACH DELETE n
                RETURN count(n) as deleted_nodes
            """, cutoff_timestamp=cutoff_timestamp)
            
            deleted_count = result.single()['deleted_nodes']
            logger.info(f"Cleaned up {deleted_count} old nodes from graph")
            
            return deleted_count
    
    def create_graph_snapshot(self, snapshot_name: str) -> Dict[str, Any]:
        """Create a named snapshot of the current graph state"""
        analytics = self.get_graph_analytics()
        graph_data = self.export_graph_data()
        
        snapshot = {
            'name': snapshot_name,
            'timestamp': time.time(),
            'analytics': analytics,
            'graph_data': graph_data
        }
        
        # Save snapshot to Neo4j as a special node
        with self.driver.session() as session:
            session.run("""
                CREATE (s:Snapshot {
                    name: $name,
                    timestamp: $timestamp,
                    data: $data
                })
            """,
            name=snapshot_name,
            timestamp=snapshot.get('timestamp'),
            data=json.dumps(snapshot)
            )
        
        logger.info(f"Created graph snapshot: {snapshot_name}")
        return snapshot
    
    def generate_visualization_config(self) -> Dict[str, Any]:
        """Generate configuration for web-based graph visualization"""
        config = {
            'layout': {
                'name': 'force-directed',
                'settings': {
                    'attraction': 0.1,
                    'repulsion': 1.0,
                    'damping': 0.9
                }
            },
            'node_styling': {
                'size_property': 'local_id',
                'color_property': 'data_type',
                'label_property': 'id',
                'size_range': [10, 50],
                'colors': {
                    'text': '#3498db',
                    'image': '#e74c3c',
                    'audio': '#2ecc71',
                    'unknown': '#95a5a6'
                }
            },
            'edge_styling': {
                'width_property': 'strength',
                'color_property': 'similarity',
                'width_range': [1, 10],
                'color_gradient': ['#bdc3c7', '#e74c3c']
            },
            'filters': {
                'min_similarity': 0.1,
                'data_types': ['text', 'image', 'audio'],
                'date_range': {
                    'start': time.time() - (30 * 24 * 60 * 60),  # 30 days ago
                    'end': time.time()
                }
            },
            'interactions': {
                'click_action': 'show_details',
                'hover_action': 'highlight_neighbors',
                'double_click_action': 'expand_neighborhood'
            }
        }
        
        return config
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
        logger.info("Neo4j visualizer closed")

def demo_neo4j_visualizer():
    """Demonstrate Neo4j visualization capabilities"""
    print("🎨 Neo4j Graph Visualization Demo")
    print("=================================")
    
    # Initialize visualizer
    visualizer = AdvancedNeo4jVisualizer()
    
    # Add sample data
    print("\n📊 Adding sample data nodes...")
    
    sample_data = [
        ("node_1", "text", 384, {"content": "Machine learning"}),
        ("node_2", "text", 384, {"content": "Deep learning"}),
        ("node_3", "image", 2048, {"filename": "cat.jpg"}),
        ("node_4", "audio", 1300, {"filename": "music.wav"}),
        ("node_5", "text", 384, {"content": "Neural networks"})
    ]
    
    for node_id, data_type, vector_dim, metadata in sample_data:
        visualizer.add_data_node(
            node_id=node_id,
            data_type=data_type,
            vector_dim=vector_dim,
            timestamp=time.time(),
            metadata=metadata,
            local_id=np.random.uniform(2.0, 8.0),
            partition_id=np.random.randint(0, 10)
        )
    
    # Add similarity relationships
    print("🔗 Adding similarity relationships...")
    similarities = [
        ("node_1", "node_2", 0.89),  # Similar text
        ("node_1", "node_5", 0.76),  # Related text
        ("node_2", "node_5", 0.82),  # Related text
        ("node_3", "node_4", 0.23),  # Different modalities
    ]
    
    for node1, node2, similarity in similarities:
        visualizer.add_similarity_relationship(node1, node2, similarity)
    
    # Get analytics
    print("\n📈 Graph Analytics:")
    analytics = visualizer.get_graph_analytics()
    
    print("  Basic Statistics:")
    basic_stats = analytics.get('basic_stats', {})
    for key, value in basic_stats.items():
        print(f"    {key}: {value}")
    
    print("  Data Type Distribution:")
    for item in analytics.get('type_distribution', []):
        print(f"    {item['data_type']}: {item['count']} nodes")
    
    # Find clusters
    print("\n🔍 Finding similarity clusters...")
    clusters = visualizer.find_similar_clusters(min_similarity=0.7, min_cluster_size=2)
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {cluster['center_node']} ({cluster['data_type']}) "
              f"with {cluster['cluster_size']} similar nodes")
    
    # Get neighborhood
    print("\n🌐 Node neighborhood analysis:")
    neighborhood = visualizer.get_node_neighborhood("node_1", max_hops=2)
    if neighborhood:
        print(f"  Center: {neighborhood['center_id']} ({neighborhood['center_type']})")
        print(f"  Neighbors: {neighborhood['neighbor_count']}")
        for neighbor in neighborhood.get('neighbors', [])[:3]:
            print(f"    - {neighbor['id']} ({neighbor['data_type']})")
    
    # Export graph data
    print("\n💾 Exporting graph data...")
    graph_data = visualizer.export_graph_data()
    print(f"  Exported {len(graph_data['nodes'])} nodes and {len(graph_data['relationships'])} edges")
    
    # Generate visualization config
    print("\n🎨 Generating visualization configuration...")
    viz_config = visualizer.generate_visualization_config()
    print(f"  Configuration includes: {list(viz_config.keys())}")
    
    # Create snapshot
    print("\n📸 Creating graph snapshot...")
    snapshot = visualizer.create_graph_snapshot("demo_snapshot")
    print(f"  Snapshot created: {snapshot['name']} at {snapshot['timestamp']}")
    
    visualizer.close()
    print("\n✅ Neo4j visualization demo completed!")

if __name__ == "__main__":
    demo_neo4j_visualizer()
