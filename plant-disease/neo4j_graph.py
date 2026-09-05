"""
Neo4j Knowledge Graph Module for Plant Disease GraphRAG
Connects to Neo4j graph database and manages semantic relationships between
Crops, Diseases, Pathogens, Symptoms, and Treatments.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "MyStrongNeo4jPassword_2026!")


class PlantNeo4jGraph:
    """
    Manages GraphRAG queries and knowledge graph synchronization in Neo4j.
    """
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._init_driver()

    def _init_driver(self):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        except Exception as e:
            print(f"[Neo4j Warning] Could not initialize driver: {e}")
            self.driver = None

    def is_connected(self) -> bool:
        """Verifies if the Neo4j database instance is running and reachable."""
        if not self.driver:
            return False
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS connected")
                record = result.single()
                return record and record["connected"] == 1
        except Exception:
            return False

    def seed_knowledge_graph(self) -> bool:
        """
        Seeds standard agricultural knowledge graph nodes and relationships into Neo4j.
        """
        if not self.is_connected():
            print(f"[Neo4j] Database at {self.uri} not reachable. Start Neo4j server to enable GraphRAG.")
            return False

        cypher_script = """
        // Constraints
        CREATE CONSTRAINT crop_name IF NOT EXISTS FOR (c:Crop) REQUIRE c.name IS UNIQUE;
        CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE;

        // Tomato Knowledge Subgraph
        MERGE (c1:Crop {name: 'Tomato', scientific_name: 'Solanum lycopersicum'})
        MERGE (d1:Disease {name: 'Early Blight', severity: 'Moderate to High'})
        MERGE (d2:Disease {name: 'Late Blight', severity: 'CRITICAL'})
        MERGE (d3:Disease {name: 'Bacterial Spot', severity: 'High'})
        
        MERGE (c1)-[:HAS_DISEASE]->(d1)
        MERGE (c1)-[:HAS_DISEASE]->(d2)
        MERGE (c1)-[:HAS_DISEASE]->(d3)

        MERGE (p1:Pathogen {name: 'Alternaria solani', type: 'Fungus'})
        MERGE (p2:Pathogen {name: 'Phytophthora infestans', type: 'Oomycete'})
        MERGE (p3:Pathogen {name: 'Xanthomonas spp.', type: 'Bacterium'})
        
        MERGE (d1)-[:CAUSED_BY]->(p1)
        MERGE (d2)-[:CAUSED_BY]->(p2)
        MERGE (d3)-[:CAUSED_BY]->(p3)

        MERGE (s1:Symptom {description: 'Concentric ring target board spots on lower foliage'})
        MERGE (s2:Symptom {description: 'Water-soaked purplish-black lesions with white fungal down'})
        MERGE (s3:Symptom {description: 'Small dark water-soaked spots with yellow halo'})
        
        MERGE (d1)-[:HAS_SYMPTOM]->(s1)
        MERGE (d2)-[:HAS_SYMPTOM]->(s2)
        MERGE (d3)-[:HAS_SYMPTOM]->(s3)

        MERGE (t1:Treatment {name: 'Liquid Copper Octanoate / Bacillus subtilis', category: 'Organic'})
        MERGE (t2:Treatment {name: 'Chlorothalonil / Mancozeb / Azoxystrobin', category: 'Chemical'})
        MERGE (t3:Treatment {name: 'Cyazofamid / Mandipropamid (Revus)', category: 'Chemical'})
        
        MERGE (d1)-[:TREATED_BY]->(t1)
        MERGE (d1)-[:TREATED_BY]->(t2)
        MERGE (d2)-[:TREATED_BY]->(t3)

        // Potato Knowledge Subgraph
        MERGE (c2:Crop {name: 'Potato', scientific_name: 'Solanum tuberosum'})
        MERGE (c2)-[:HAS_DISEASE]->(d1)
        MERGE (c2)-[:HAS_DISEASE]->(d2)

        // Rice Knowledge Subgraph
        MERGE (c3:Crop {name: 'Rice', scientific_name: 'Oryza sativa'})
        MERGE (d4:Disease {name: 'Leaf Blast', severity: 'CRITICAL'})
        MERGE (c3)-[:HAS_DISEASE]->(d4)
        MERGE (p4:Pathogen {name: 'Magnaporthe oryzae', type: 'Fungus'})
        MERGE (d4)-[:CAUSED_BY]->(p4)
        MERGE (t4:Treatment {name: 'Tricyclazole 75% WP / Kasugamycin', category: 'Chemical'})
        MERGE (d4)-[:TREATED_BY]->(t4)
        """

        try:
            with self.driver.session() as session:
                for statement in cypher_script.strip().split(";"):
                    stmt = statement.strip()
                    if stmt:
                        session.run(stmt)
            print("[Neo4j] Knowledge graph successfully seeded with Crop, Disease, and Treatment entities.")
            return True
        except Exception as e:
            print(f"[Neo4j Error] Seeding graph failed: {e}")
            return False

    def query_disease_relations(self, crop: str, disease: str) -> Dict[str, Any]:
        """
        Retrieves graph neighbors for a given crop and disease.
        """
        if not self.is_connected():
            return {"connected": False, "nodes": [], "message": "Neo4j database not reachable at bolt://localhost:7687"}

        query = """
        MATCH (c:Crop)-[:HAS_DISEASE]->(d:Disease)
        WHERE toLower(c.name) CONTAINS toLower($crop) AND toLower(d.name) CONTAINS toLower($disease)
        OPTIONAL MATCH (d)-[:CAUSED_BY]->(p:Pathogen)
        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
        RETURN c.name AS crop, d.name AS disease, d.severity AS severity,
               collect(DISTINCT p.name) AS pathogens,
               collect(DISTINCT s.description) AS symptoms,
               collect(DISTINCT t.name) AS treatments
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, crop=crop, disease=disease)
                record = result.single()
                if record:
                    return {
                        "connected": True,
                        "crop": record["crop"],
                        "disease": record["disease"],
                        "severity": record["severity"],
                        "pathogens": record["pathogens"],
                        "symptoms": record["symptoms"],
                        "treatments": record["treatments"]
                    }
        except Exception as e:
            print(f"[Neo4j Query Error] {e}")

        return {"connected": True, "message": "No matching graph entities found for this query."}

    def close(self):
        if self.driver:
            self.driver.close()


_neo4j_instance = None

def get_neo4j_graph() -> PlantNeo4jGraph:
    global _neo4j_instance
    if _neo4j_instance is None:
        _neo4j_instance = PlantNeo4jGraph()
    return _neo4j_instance


if __name__ == "__main__":
    graph = get_neo4j_graph()
    connected = graph.is_connected()
    print(f"Neo4j Connection ({NEO4J_URI}): {'CONNECTED' if connected else 'NOT REACHABLE (Local service inactive)'}")
    if connected:
        graph.seed_knowledge_graph()
        rel = graph.query_disease_relations("Tomato", "Early Blight")
        print("Neo4j Query Result:", rel)
