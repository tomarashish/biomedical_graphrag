"""
EFO to Neo4j - Create a Neo4j graph from the EFO (Experimental Factor Ontology)

This script downloads the EFO ontology, parses it, and creates a graph in Neo4j.
The EFO ontology provides a systematic description of experimental variables available in EBI databases.

Requirements:
- pronto (for parsing OBO files)
- neo4j (for connecting to Neo4j)
- requests (for downloading the ontology)
"""

import os
import requests
import neo4j
from pronto import Ontology
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7693')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

# EFO ontology URL
EFO_URL = "https://www.ebi.ac.uk/efo/efo.obo"
EFO_FILE = "efo.obo"

def download_efo_ontology(url=EFO_URL, file_path=EFO_FILE):
    """
    Download the EFO ontology file if it doesn't exist locally
    
    Args:
        url (str): URL to download the ontology from
        file_path (str): Path to save the ontology file
    
    Returns:
        str: Path to the downloaded file
    """
    if not os.path.exists(file_path):
        logger.info(f"Downloading EFO ontology from {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded EFO ontology to {file_path}")
    else:
        logger.info(f"EFO ontology file already exists at {file_path}")
    
    return file_path

def parse_efo_ontology(file_path=EFO_FILE):
    """
    Parse the EFO ontology file
    
    Args:
        file_path (str): Path to the ontology file
    
    Returns:
        Ontology: Parsed ontology object
    """
    logger.info(f"Parsing EFO ontology from {file_path}")
    ontology = Ontology(file_path)
    logger.info(f"Parsed {len(ontology.terms())} terms from the ontology")
    return ontology

def create_neo4j_constraints(driver):
    """
    Create Neo4j constraints for the EFO graph
    
    Args:
        driver: Neo4j driver instance
    """
    logger.info("Creating Neo4j constraints")
    
    with driver.session(database=NEO4J_DATABASE) as session:
        # Create constraints for unique IDs
        session.run("CREATE CONSTRAINT efo_term_id IF NOT EXISTS FOR (t:EFOTerm) REQUIRE t.id IS UNIQUE")
        session.run("CREATE CONSTRAINT efo_relationship_id IF NOT EXISTS FOR ()-[r:EFO_RELATIONSHIP]-() REQUIRE r.id IS UNIQUE")

def create_efo_graph(driver, ontology):
    """
    Create a Neo4j graph from the EFO ontology
    
    Args:
        driver: Neo4j driver instance
        ontology: Parsed ontology object
    """
    logger.info("Creating EFO graph in Neo4j")
    
    with driver.session(database=NEO4J_DATABASE) as session:
        # First, clear any existing EFO data
        session.run("MATCH (n:EFOTerm) DETACH DELETE n")
        
        # Create nodes for each term
        term_count = 0
        for term_id, term in ontology.terms.items():
            # Skip terms without an ID
            if not term_id:
                continue
                
            # Create properties for the term
            properties = {
                "id": term_id,
                "name": term.name or "",
                "definition": term.definition or "",
                "comment": term.comment or "",
                "namespace": term.namespace or ""
            }
            
            # Add synonyms if available
            if hasattr(term, 'synonyms') and term.synonyms:
                properties["synonyms"] = [syn.description for syn in term.synonyms]
            
            # Create the term node
            session.run("""
                MERGE (t:EFOTerm {id: $id})
                SET t.name = $name,
                    t.definition = $definition,
                    t.comment = $comment,
                    t.namespace = $namespace,
                    t.synonyms = $synonyms
            """, properties)
            
            term_count += 1
            if term_count % 1000 == 0:
                logger.info(f"Created {term_count} term nodes")
        
        logger.info(f"Created a total of {term_count} term nodes")
        
        # Create relationships between terms
        rel_count = 0
        for term_id, term in ontology.terms.items():
            # Skip terms without an ID
            if not term_id:
                continue
                
            # Process relationships
            for relationship in term.relationships:
                rel_type = relationship.type.id
                target_id = relationship.target.id
                
                # Create the relationship
                session.run("""
                    MATCH (source:EFOTerm {id: $source_id})
                    MATCH (target:EFOTerm {id: $target_id})
                    MERGE (source)-[r:EFO_RELATIONSHIP {type: $rel_type}]->(target)
                    SET r.id = $source_id + '_' + $rel_type + '_' + $target_id
                """, {
                    "source_id": term_id,
                    "target_id": target_id,
                    "rel_type": rel_type
                })
                
                rel_count += 1
                if rel_count % 1000 == 0:
                    logger.info(f"Created {rel_count} relationships")
        
        logger.info(f"Created a total of {rel_count} relationships")
        
        # Process subclass relationships (is_a)
        subclass_count = 0
        for term_id, term in ontology.terms.items():
            # Skip terms without an ID
            if not term_id:
                continue
                
            # Process subclass relationships
            for parent in term.superclasses():
                parent_id = parent.id
                
                # Create the is_a relationship
                session.run("""
                    MATCH (child:EFOTerm {id: $child_id})
                    MATCH (parent:EFOTerm {id: $parent_id})
                    MERGE (child)-[r:IS_A]->(parent)
                    SET r.id = $child_id + '_IS_A_' + $parent_id
                """, {
                    "child_id": term_id,
                    "parent_id": parent_id
                })
                
                subclass_count += 1
                if subclass_count % 1000 == 0:
                    logger.info(f"Created {subclass_count} is_a relationships")
        
        logger.info(f"Created a total of {subclass_count} is_a relationships")

def main():
    """
    Main function to execute the EFO to Neo4j conversion
    """
    try:
        # Download the EFO ontology
        efo_file = download_efo_ontology()
        
        # Parse the ontology
        ontology = parse_efo_ontology(efo_file)
        
        # Connect to Neo4j
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
        driver = neo4j.GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        
        # Test the connection
        try:
            driver.verify_connectivity()
            logger.info("Neo4j connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return
        
        # Create constraints
        create_neo4j_constraints(driver)
        
        # Create the EFO graph
        create_efo_graph(driver, ontology)
        
        # Close the driver
        driver.close()
        logger.info("EFO to Neo4j conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Error in EFO to Neo4j conversion: {e}")
        raise

if __name__ == "__main__":
    main()
