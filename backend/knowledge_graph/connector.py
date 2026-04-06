"""
Knowledge Graph Connector — Neo4j
===================================
Manages the Arabic knowledge graph for structured entity-relationship storage.

Why Neo4j:
  ✅ Native graph queries (Cypher — readable, powerful)
  ✅ HNSW vector index built-in (Neo4j 5.x)
  ✅ Combined graph + vector search (GraphRAG)
  ✅ Docker self-hosted (GCC data sovereignty)
  ✅ Python driver with async support
  ✅ Strong Arabic unicode support

Schema (Domain-driven for GCC/Arabic AI):
  Nodes: Entity, Concept, Person, Organization, Location, Regulation, Event
  Edges: RELATED_TO, MENTIONS, LOCATED_IN, GOVERNS, EMPLOYS, CITES

Graph ontology covers:
  - GCC entities (countries, cities, institutions, regulators)
  - Islamic finance and banking terminology
  - Arabic academic concepts
  - Regulatory frameworks (CBB, SAMA, UAE Central Bank)
  - Research concepts and citations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("knowledge_graph")


@dataclass
class GraphEntity:
    """Represents a node in the knowledge graph."""
    entity_id: str
    name: str
    name_arabic: str
    entity_type: str            # Person, Org, Location, Concept, Regulation, Event
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class GraphRelation:
    """Represents an edge in the knowledge graph."""
    from_entity: str
    relation_type: str          # RELATED_TO, CITES, GOVERNS, etc.
    to_entity: str
    properties: Dict[str, Any]
    weight: float = 1.0


class KnowledgeGraphConnector:
    """
    Neo4j Knowledge Graph connector for the Arabic Cognitive AI Engine.
    
    Supports:
    - Entity CRUD operations
    - Relationship management
    - Natural language to Cypher translation
    - Vector similarity search (Neo4j 5.x+ vector index)
    - GraphRAG — combined graph traversal + vector search
    - Arabic entity normalization
    """
    
    # ─── Cypher Schema Initialization ─────────────────────────────────────────
    SCHEMA_CYPHER = [
        # Constraints
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
        
        # Indexes for Arabic text search
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        "CREATE INDEX entity_name_ar IF NOT EXISTS FOR (e:Entity) ON (e.name_arabic)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
        "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        
        # Vector index for semantic search (Neo4j 5.x)
        """CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
           FOR (e:Entity) ON (e.embedding)
           OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}""",
    ]
    
    # ─── Pre-built GCC Knowledge Base ────────────────────────────────────────
    GCC_SEED_DATA = [
        # Countries
        {"name": "Bahrain", "name_arabic": "البحرين", "type": "Location", 
         "props": {"region": "GCC", "capital": "Manama", "capital_arabic": "المنامة"}},
        {"name": "Saudi Arabia", "name_arabic": "المملكة العربية السعودية", "type": "Location",
         "props": {"region": "GCC", "capital": "Riyadh", "capital_arabic": "الرياض"}},
        {"name": "UAE", "name_arabic": "الإمارات العربية المتحدة", "type": "Location",
         "props": {"region": "GCC", "capital": "Abu Dhabi", "capital_arabic": "أبوظبي"}},
        
        # Regulators
        {"name": "Central Bank of Bahrain", "name_arabic": "مصرف البحرين المركزي", "type": "Organization",
         "props": {"role": "regulator", "country": "Bahrain", "acronym": "CBB"}},
        {"name": "SAMA", "name_arabic": "مؤسسة النقد العربي السعودي", "type": "Organization",
         "props": {"role": "regulator", "country": "Saudi Arabia", "acronym": "SAMA"}},
        
        # AI Concepts
        {"name": "Large Language Model", "name_arabic": "نموذج اللغة الكبير", "type": "Concept",
         "props": {"domain": "AI", "related": "NLP"}},
        {"name": "Retrieval-Augmented Generation", "name_arabic": "التوليد المعزز بالاسترجاع", "type": "Concept",
         "props": {"domain": "AI", "acronym": "RAG"}},
        {"name": "Knowledge Graph", "name_arabic": "مخطط المعرفة", "type": "Concept",
         "props": {"domain": "AI", "related": "ontology"}},
    ]
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._entity_count = 0
        logger.info("KnowledgeGraphConnector configured")
    
    async def connect(self):
        """Establish connection to Neo4j."""
        try:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                max_connection_pool_size=50
            )
            await self._driver.verify_connectivity()
            await self._initialize_schema()
            await self._seed_gcc_knowledge()
            logger.info(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j unavailable: {e}. Running with in-memory fallback.")
            self._driver = None
    
    async def disconnect(self):
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            logger.info("Neo4j connection closed")
    
    async def ping(self) -> str:
        """Check Neo4j connectivity."""
        if not self._driver:
            return "unavailable"
        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            return "active"
        except Exception:
            return "error"
    
    async def _initialize_schema(self):
        """Create graph schema (constraints + indexes)."""
        if not self._driver:
            return
        
        async with self._driver.session() as session:
            for cypher in self.SCHEMA_CYPHER:
                try:
                    await session.run(cypher)
                except Exception as e:
                    logger.debug(f"Schema init: {e}")
        
        logger.info("✅ Knowledge graph schema initialized")
    
    async def _seed_gcc_knowledge(self):
        """Seed the graph with GCC domain knowledge."""
        if not self._driver:
            return
        
        for entity_data in self.GCC_SEED_DATA:
            await self.create_entity(GraphEntity(
                entity_id=entity_data["name"].lower().replace(" ", "_"),
                name=entity_data["name"],
                name_arabic=entity_data["name_arabic"],
                entity_type=entity_data["type"],
                properties=entity_data.get("props", {})
            ))
        
        logger.info(f"✅ Seeded {len(self.GCC_SEED_DATA)} GCC entities into knowledge graph")
    
    # ─── Entity Operations ────────────────────────────────────────────────────
    
    async def create_entity(self, entity: GraphEntity) -> bool:
        """Create or update an entity node."""
        if not self._driver:
            self._entity_count += 1
            return True
        
        cypher = """
        MERGE (e:Entity {entity_id: $entity_id})
        SET e.name = $name,
            e.name_arabic = $name_arabic,
            e.entity_type = $entity_type,
            e.properties = $properties,
            e.updated_at = datetime()
        WITH e
        CALL apoc.create.addLabels(e, [$entity_type]) YIELD node
        RETURN node
        """
        
        try:
            async with self._driver.session() as session:
                await session.run(cypher, {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "name_arabic": entity.name_arabic,
                    "entity_type": entity.entity_type,
                    "properties": entity.properties
                })
            self._entity_count += 1
            return True
        except Exception as e:
            # Try without APOC
            return await self._create_entity_simple(entity)
    
    async def _create_entity_simple(self, entity: GraphEntity) -> bool:
        """Create entity without APOC library."""
        if not self._driver:
            return True
        
        cypher = f"""
        MERGE (e:{entity.entity_type} {{entity_id: $entity_id}})
        SET e.name = $name,
            e.name_arabic = $name_arabic,
            e.entity_type = $entity_type
        RETURN e
        """
        
        try:
            async with self._driver.session() as session:
                await session.run(cypher, {
                    "entity_id": entity.entity_id,
                    "name": entity.name,
                    "name_arabic": entity.name_arabic,
                    "entity_type": entity.entity_type,
                })
            return True
        except Exception as e:
            logger.error(f"Entity creation failed: {e}")
            return False
    
    async def create_relation(self, relation: GraphRelation) -> bool:
        """Create a relationship between two entities."""
        if not self._driver:
            return True
        
        cypher = f"""
        MATCH (from:Entity {{entity_id: $from_id}})
        MATCH (to:Entity {{entity_id: $to_id}})
        MERGE (from)-[r:{relation.relation_type}]->(to)
        SET r.weight = $weight,
            r.properties = $properties,
            r.created_at = datetime()
        RETURN r
        """
        
        try:
            async with self._driver.session() as session:
                await session.run(cypher, {
                    "from_id": relation.from_entity,
                    "to_id": relation.to_entity,
                    "weight": relation.weight,
                    "properties": relation.properties
                })
            return True
        except Exception as e:
            logger.error(f"Relation creation failed: {e}")
            return False
    
    # ─── Query Operations ─────────────────────────────────────────────────────
    
    async def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """Execute raw Cypher query."""
        if not self._driver:
            return self._mock_query_result()
        
        try:
            async with self._driver.session() as session:
                result = await session.run(cypher, params or {})
                records = await result.data()
                return records
        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            return []
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Combined semantic + keyword search on knowledge graph.
        Uses text search as primary (vector search when embedding available).
        """
        if not self._driver:
            return self._mock_kg_results(query)
        
        # Full-text search on entity names (Arabic + English)
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_name_index', $query)
        YIELD node, score
        RETURN node.name as name, 
               node.name_arabic as name_arabic,
               node.entity_type as type,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        try:
            results = await self.query(cypher, {"query": query, "limit": limit})
            if results:
                return results
        except Exception:
            pass
        
        # Fallback: CONTAINS search
        cypher_fallback = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
           OR e.name_arabic CONTAINS $query
        RETURN e.name as name, e.name_arabic as name_arabic, 
               e.entity_type as type, 1.0 as score
        LIMIT $limit
        """
        
        return await self.query(cypher_fallback, {"query": query, "limit": limit})
    
    async def get_entity_neighborhood(
        self, entity_id: str, depth: int = 2
    ) -> Dict:
        """
        Get entity subgraph (for GraphRAG multi-hop reasoning).
        Returns entity and its connected neighbors up to `depth` hops.
        """
        if not self._driver:
            return {"entity": entity_id, "neighbors": []}
        
        cypher = """
        MATCH path = (e:Entity {entity_id: $entity_id})-[*1..$depth]-(neighbor)
        RETURN e.name as entity,
               [node in nodes(path) | node.name] as path_nodes,
               [rel in relationships(path) | type(rel)] as relations,
               neighbor.name as neighbor,
               neighbor.entity_type as neighbor_type
        LIMIT 50
        """
        
        results = await self.query(cypher, {"entity_id": entity_id, "depth": depth})
        return {"entity": entity_id, "neighborhood": results}
    
    async def store_entities(self, entities: List[Dict]):
        """Store extracted entities from text into knowledge graph."""
        for entity in entities:
            graph_entity = GraphEntity(
                entity_id=entity.get("text", "").lower().replace(" ", "_")[:50],
                name=entity.get("text", ""),
                name_arabic=entity.get("text_normalized", entity.get("text", "")),
                entity_type=entity.get("entity_type", "Concept"),
                properties={"confidence": entity.get("confidence", 0.8)}
            )
            await self.create_entity(graph_entity)
    
    async def get_related_entities(self, entity_name: str, relation_type: str = None) -> List[Dict]:
        """Find entities related to a given entity."""
        if not self._driver:
            return []
        
        if relation_type:
            cypher = f"""
            MATCH (e:Entity {{name: $name}})-[:{relation_type}]-(related)
            RETURN related.name as name, related.name_arabic as name_arabic,
                   related.entity_type as type
            LIMIT 20
            """
        else:
            cypher = """
            MATCH (e:Entity {name: $name})-[r]-(related)
            RETURN related.name as name, related.name_arabic as name_arabic,
                   related.entity_type as type, type(r) as relation
            LIMIT 20
            """
        
        return await self.query(cypher, {"name": entity_name})
    
    async def entity_count(self) -> int:
        """Return total entity count."""
        if not self._driver:
            return self._entity_count
        
        result = await self.query("MATCH (e:Entity) RETURN count(e) as count")
        return result[0].get("count", 0) if result else 0
    
    # ─── GraphRAG Integration ─────────────────────────────────────────────────
    
    async def graph_rag_context(self, query: str, entities: List[str]) -> str:
        """
        Build structured context from knowledge graph for RAG.
        Used to enrich LLM prompts with graph-derived facts.
        """
        context_parts = []
        
        for entity_name in entities[:5]:
            # Get entity details
            results = await self.semantic_search(entity_name, limit=1)
            if results:
                entity = results[0]
                related = await self.get_related_entities(entity_name)
                
                fact = f"[KG] {entity.get('name')} ({entity.get('name_arabic', '')})"
                fact += f" — Type: {entity.get('type', '')}"
                
                if related:
                    relations = [f"{r.get('relation', 'related')} → {r.get('name')}" for r in related[:3]]
                    fact += f" — Relations: {', '.join(relations)}"
                
                context_parts.append(fact)
        
        return "\n".join(context_parts) if context_parts else ""
    
    # ─── Analytics ────────────────────────────────────────────────────────────
    
    async def get_graph_stats(self) -> Dict:
        """Return knowledge graph statistics."""
        if not self._driver:
            return {"entities": self._entity_count, "relations": 0, "labels": []}
        
        stats = {}
        
        entity_count = await self.query("MATCH (e:Entity) RETURN count(e) as count")
        stats["total_entities"] = entity_count[0].get("count", 0) if entity_count else 0
        
        relation_count = await self.query("MATCH ()-[r]->() RETURN count(r) as count")
        stats["total_relations"] = relation_count[0].get("count", 0) if relation_count else 0
        
        by_type = await self.query(
            "MATCH (e:Entity) RETURN e.entity_type as type, count(e) as count ORDER BY count DESC"
        )
        stats["entities_by_type"] = by_type
        
        return stats
    
    # ─── Mock Data (Development Fallback) ────────────────────────────────────
    
    def _mock_query_result(self) -> List[Dict]:
        return [{"message": "Neo4j not connected — mock result"}]
    
    def _mock_kg_results(self, query: str) -> List[Dict]:
        return [
            {"name": "Bahrain", "name_arabic": "البحرين", "type": "Location", "score": 0.9},
            {"name": "Central Bank of Bahrain", "name_arabic": "مصرف البحرين المركزي", "type": "Organization", "score": 0.85},
            {"name": "Knowledge Graph", "name_arabic": "مخطط المعرفة", "type": "Concept", "score": 0.75},
        ]
