import os
import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz, chisq, kci, gsq
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from collections import defaultdict


class StructuralLearningEngine:
    def __init__(self, feature_names):
        """Initialize the structural learning framework with feature mapping"""
        print("Setting up Structural Learning Engine")      
        self.feature_mapping = {idx: feature_names[idx] for idx in range(len(feature_names))}    

    def configure_domain_constraints(self, graph_obj, blocked_connections=None, forced_connections=None):
        """Establishes domain-specific constraints for structural learning algorithm.

        Parameters:
            graph_obj (Graph): The structural graph from learning process.
            blocked_connections (list of tuples): Prohibited connections (e.g., [('A', 'B'), ('C', 'D')]).
            forced_connections (list of tuples): Mandatory connections (e.g., [('X', 'Y'), ('A', 'C')]).

        Returns:
            domain_knowledge (BackgroundKnowledge): Constraint object with domain restrictions.
        """
        domain_knowledge = BackgroundKnowledge()
        graph_nodes = graph_obj.get_nodes()  # Extract nodes from the structural graph

        if blocked_connections:
            for connection in blocked_connections:
                if connection[0] in self.feature_mapping.values() and connection[1] in self.feature_mapping.values():
                    source_position = list(self.feature_mapping.values()).index(connection[0])
                    target_position = list(self.feature_mapping.values()).index(connection[1])
                    source_vertex = graph_nodes[source_position]  # Map to actual graph vertices
                    target_vertex = graph_nodes[target_position]
                    domain_knowledge.add_forbidden_by_node(source_vertex, target_vertex)  # Apply restriction

        if forced_connections:
            for connection in forced_connections:
                if connection[0] in self.feature_mapping.values() and connection[1] in self.feature_mapping.values():
                    source_position = list(self.feature_mapping.values()).index(connection[0])
                    target_position = list(self.feature_mapping.values()).index(connection[1])
                    source_vertex = graph_nodes[source_position]  # Map to actual graph vertices
                    target_vertex = graph_nodes[target_position]
                    domain_knowledge.add_required_by_node(source_vertex, target_vertex)  # Enforce connection

        return domain_knowledge

    def execute_constrained_discovery(self, dataset, blocked_connections, forced_connections, significance_threshold):
        """Executes constrained structural discovery using domain knowledge.

        Parameters:
            dataset (pd.DataFrame): Input observations.
            blocked_connections (list of tuples): Prohibited connections.
            forced_connections (list of tuples): Mandatory connections.
            significance_threshold (float): Statistical significance level for independence tests.

        Returns:
            discovered_structure (list), learned_graph (Graph): The inferred structural relationships and graph.
        """
        data_matrix = np.array(dataset)  # Transform DataFrame to matrix format

        # Initial structural learning pass to obtain graph structure
        learned_graph, edge_set = fci(data_matrix, chisq, alpha=significance_threshold, verbose=True)

        # Configure domain constraints based on initial structure
        domain_knowledge = self.configure_domain_constraints(learned_graph, blocked_connections, forced_connections)

        # Execute constrained structural learning with domain knowledge
        learned_graph, edge_set = fci(data_matrix, chisq, alpha=significance_threshold, verbose=True, background_knowledge=domain_knowledge)

        discovered_structure = [str(relationship) for relationship in edge_set]  # Convert to string representation

        print("Discovered structural relationships with constraints:", discovered_structure)

        # Create output directory for visualizations
        visualization_dir = "fig"
        os.makedirs(visualization_dir, exist_ok=True)  

        # Generate and save structural graph visualization
        graph_diagram = GraphUtils.to_pydot(learned_graph, edge_set, labels=dataset.columns.to_list())          
        graph_diagram.write_png(os.path.join(visualization_dir, 'discovered_graph.png'))

        return discovered_structure, learned_graph
    
    def interpret_structural_relationships(self, baseline_structure, discovered_structure, feature_names, prohibited_links, target_variables, path_limit):
        """Interprets and processes discovered structural relationships"""
        bidirectional_link = "<->"
        unidirectional_link = "-->"
        undetermined_link = "o-o"
        partial_link = "o->"
        
        # Initialize relationship storage for each feature
        relationship_options = {}
        for feature in feature_names:
            relationship_options[feature] = {}
            relationship_options[feature][unidirectional_link] = []
            relationship_options[feature][bidirectional_link] = []
            
        # Incorporate baseline structural knowledge
        for relationship in baseline_structure:
            if relationship[0] or relationship[1] is None:
                relationship_options[relationship[0]][unidirectional_link].append(relationship[1])
                
        # Standardize uncertain relationships using heuristic approach
        for idx in range(len(discovered_structure)):
            if partial_link in discovered_structure[idx]:
                discovered_structure[idx] = discovered_structure[idx].replace(partial_link, unidirectional_link)
            elif undetermined_link in discovered_structure[idx]:
                discovered_structure[idx] = discovered_structure[idx].replace(undetermined_link, unidirectional_link)
            else:
                continue
        
        # Process discovered relationships into structured format
        for relationship in discovered_structure:
            components = relationship.split(" ")
            if components[1] == unidirectional_link:
                source_feature = self.feature_mapping[int(components[0].replace("X", ""))-1]               
                target_feature = self.feature_mapping[int(components[2].replace("X", ""))-1]
                relationship_options[source_feature][unidirectional_link].append(target_feature)
            elif components[1] == bidirectional_link:
                source_feature = self.feature_mapping[int(components[0].replace("X", ""))-1]
                target_feature = self.feature_mapping[int(components[2].replace("X", ""))-1]
                relationship_options[source_feature][bidirectional_link].append(target_feature)
            else: 
                print("[WARNING]: Unexpected relationship type encountered")
                
        # Extract different types of structural relationships
        unidirectional_relationships = []
        bidirectional_relationships = []
        
        for feature in relationship_options:
            relationship_options[feature][unidirectional_link] = list(set(relationship_options[feature][unidirectional_link]))
            relationship_options[feature][bidirectional_link] = list(set(relationship_options[feature][bidirectional_link]))
            
        for feature in relationship_options:
            for target in relationship_options[feature][unidirectional_link]:
                unidirectional_relationships.append((feature, target))
            for target in relationship_options[feature][bidirectional_link]:
                bidirectional_relationships.append((feature, target))
                
        # Filter out prohibited relationships
        filtered_unidirectional = list(set(unidirectional_relationships) - set(prohibited_links))
        clean_unidirectional = []
        for relationship in filtered_unidirectional: 
            if relationship[0] != relationship[1]:  # Remove self-loops
                clean_unidirectional.append(relationship)
        
        # Enhance with target-specific relationships
        for idx in range(int(len(filtered_unidirectional)/2)):
            for target_var in target_variables:
                if filtered_unidirectional[idx][0] != filtered_unidirectional[idx][1]:
                    clean_unidirectional.append((filtered_unidirectional[idx][0], target_var))
       
        bidirectional_relationships = list(set(bidirectional_relationships) - set(prohibited_links))
        
        print("=" * 60)
        print("Structural relationships identified by the learning algorithm")
        print(clean_unidirectional)
        print("=" * 60)
        
        return clean_unidirectional, bidirectional_relationships

    def extract_influence_pathways(self, feature_names, unidirectional_relations,
                         bidirectional_relations, target_variables):
        """Discovers influence pathways leading to target variables"""
        pathway_graph = PathwayGraph(feature_names)
        influence_pathways = {}
        
        for relation in unidirectional_relations:
            pathway_graph.establish_connection(relation[1], relation[0])
     
        for relation in bidirectional_relations:
            pathway_graph.establish_connection(relation[1], relation[0])
            
        for target in target_variables:
            pathway_graph.discover_all_pathways(target)
            influence_pathways[target] = pathway_graph.pathway_collection
    
        return influence_pathways 


class PathwayGraph:
    def __init__(self, vertices):
        # Number of vertices in the pathway graph
        self.vertex_count = vertices
        # Storage structure for graph connections
        self.connection_map = defaultdict(list)

    def establish_connection(self, source, destination):
        self.connection_map[source].append(destination)

    def discover_pathways_recursive(self, current_vertex, visited_status, current_pathway):
        visited_status[current_vertex] = True
        current_pathway.append(current_vertex)
        
        # Terminal condition: no outgoing connections
        if self.connection_map[current_vertex] == []:
            try:
                if self.pathway_collection:
                    self.pathway_collection.append(current_pathway[:])
            except AttributeError:
                self.pathway_collection = [current_pathway[:]]
        else:
            for connected_vertex in self.connection_map[current_vertex]:
                if visited_status[connected_vertex] == False:
                    self.discover_pathways_recursive(connected_vertex, visited_status, current_pathway)

        # Backtrack: remove current vertex and mark as unvisited
        current_pathway.pop()
        visited_status[current_vertex] = False

    def discover_all_pathways(self, starting_vertex):
        # Initialize all vertices as unvisited
        visited_status = {}
        for vertex in self.vertex_count:
            visited_status[vertex] = False
            
        # Initialize pathway storage
        current_pathway = []
        
        # Execute recursive pathway discovery
        self.discover_pathways_recursive(starting_vertex, visited_status, current_pathway)


# =============================================
# Data Loading and Processing
# =============================================
dataset = pd.read_csv("results_dataset.csv") #    results_dataset_koboki.csv

feature_columns = [
    "Global_Planner", "Controller", "Cost_Scaling_Factor", "Inflation_Radius",
    "Global_Path_Score", "Footprint_Type", "Collision",
     "Local_Path_Score" , "Relaxed_Task_Result", "Min_Global_Dist_To_Obst", "Min_Local_Dist_To_Obstacl"
] # , 
dataset = dataset[feature_columns]  

# =============================================
# Initialize Structural Learning Framework
# =============================================
learning_engine = StructuralLearningEngine(feature_columns)

# =============================================
# Specify Domain Constraints (Expert Knowledge)
# =============================================
prohibited_relationships = [
#     ("Collision", "Footprint_Type"),
# ("Collision","Inflation_Radius"),
          
]

mandatory_relationships = [
    # ("Global_Planner", "Global_Path_Score"),  
    # ("Controller", "Local_Path_Score"),
    # ("Controller", "Collision"),
    # ("Footprint_Type", "Collision"),
    # ("Footprint_Type", "Local_Path_Score"),
    # ("Cost_Scaling_Factor", "Global_Path_Score"),
    # ("Cost_Scaling_Factor", "Local_Path_Score"),
    # ("Min_Global_Dist_To_Obst", "Global_Path_Score"),
    # ("Min_Local_Dist_To_Obstacl", "Local_Path_Score"),
    # ("Collision", "Relaxed_Task_Result"),
]

# =============================================
# Execute Constrained Structural Discovery
# =============================================
structural_relationships, learned_structure = learning_engine.execute_constrained_discovery(
    dataset, prohibited_relationships, mandatory_relationships, significance_threshold=0.35
)

baseline_edges = []
outcome_variables = ["Collision"]
PATHWAY_LIMIT = 1

unidirectional_relations, bidirectional_relations = learning_engine.interpret_structural_relationships(
    baseline_edges, structural_relationships, feature_columns,
    prohibited_relationships, outcome_variables, PATHWAY_LIMIT
)

print("Bidirectional structural relationships:", bidirectional_relations)

# =============================================
# Generate Final Structural Model
# =============================================
final_structural_model = learned_structure
print("Final Structural Model (Matrix Representation):")
print(np.array(final_structural_model))

# =============================================
# Extract Structural Information
# =============================================
all_features = set(feature_columns)  
discovered_directed_relations = set(unidirectional_relations)  # Set of (source, target) pairs
latent_confounding_relations = set()  # For hidden common causes

print("Discovered directed relationships:", discovered_directed_relations)
print("Discovered directed relationships:", latent_confounding_relations)



#### # ============================================= PC Algorithm  =============================================
# from causallearn.search.ConstraintBased.PC import pc
# from causallearn.utils.cit import fisherz, chisq, kci
# import numpy as np

# # Your data matrix (n_samples x n_features)
# data_matrix = dataset.to_numpy()

# # Call PC algorithm
# cg = pc(data_matrix, 
#         alpha=0.05,  # significance level
#         indep_test=kci,  # independence test (fisherz, chisq, kci, etc.)
#         stable=True,  # use stable PC variant
#         uc_rule=0,  # orientation rule (0, 1, 2, or 3)
#         uc_priority=2,  # priority for conflicting orientations
#         mvpc=False,  # missing value PC
#         correction_name='MV_Crtn_Fisher_Z',  # for missing values
#         background_knowledge=None,  # optional background knowledge
#         verbose=False,  # print progress
#         show_progress=True,
#         node_names=feature_columns)  # show progress bar

# # Access results
# print("Causal graph:")
# print(cg.G.graph)  # adjacency matrix

# # Visualize
# from causallearn.utils.GraphUtils import GraphUtils
# import matplotlib.pyplot as plt

# pyd = GraphUtils.to_pydot(cg.G)
# pyd.write_png('causal_graph.png')
