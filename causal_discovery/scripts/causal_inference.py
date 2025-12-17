import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestRegressor


# NEW IMPORTS for additional baselines
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import cma

from itertools import product, combinations
import time
import warnings
warnings.filterwarnings('ignore')


class EnhancedCausalModel:
    def __init__(self, df, directed_edges):
        self.df = df
        self.directed_edges = directed_edges
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(directed_edges)
        self.model = None
        self.inference = None
        
        # Cache for intervened models
        self._intervention_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Cache for ACE rankings
        self._ace_ranking_cache = {}

        # Initialize variable mappings
        self.variable_mappings = {
            "Inflation_Radius": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "type": "discrete",
            },
            "Cost_Scaling_Factor": {
                "values": [0.5, 1.5, 6, 10, 20, 25, 30, 50],
                "type": "discrete",
            },
            "Footprint_Type": {"values": [0, 1], "type": "binary"},
            "Controller": {"values": [1, 2], "type": "binary"},
            "Global_Planner": {"values": [0, 1, 2, 3], "type": "discrete"},
            "Global_Path_Score": {"values": [0, 1], "type": "continuous"},
            "Local_Path_Score": {"values": [0, 1], "type": "continuous"},
            "Collision": {"values": [0, 1], "type": "binary"},
            "Min_Global_Dist_To_Obst": {"values": None, "type": "continuous"},
            "Min_Local_Dist_To_Obstacl": {"values": None, "type": "continuous"},
            "Relaxed_Task_Result": {"values": [0, 1], "type": "binary"},
            "Task_result": {"values": [0, 1], "type": "binary"},
        }

        self.original_df = df.copy()
        self.discretize_continuous_variables()
        
    def discretize_continuous_variables(self, n_bins=5):
        """Discretize continuous variables for Bayesian Network modeling"""
        self.df_discrete = self.df.copy()
        self.discretizers = {}
        
        for var, info in self.variable_mappings.items():
            if info["type"] == "continuous" and var in self.df.columns:
                discretizer = KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="quantile"
                )
                values = self.df[var].values.reshape(-1, 1)
                ordinal_labels = discretizer.fit_transform(values).flatten()
                
                bin_edges = discretizer.bin_edges_[0]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                self.df_discrete[var] = bin_centers[ordinal_labels.astype(int)]
                
                self.discretizers[var] = discretizer
                self.variable_mappings[var]["discrete_values"] = sorted(
                    self.df_discrete[var].unique()
                )

    def build_bayesian_network(self):
        """Build and validate the Bayesian Network"""
        self.model = BayesianNetwork(self.directed_edges)
        self.model.fit(self.df_discrete, estimator=MaximumLikelihoodEstimator)
        self.inference = VariableElimination(self.model)
        return self.model

    def query_probability(self, query_vars, evidence=None):
        """Query probability from the Bayesian Network"""
        if self.inference is None:
            raise ValueError("Bayesian Network not built. Call build_bayesian_network() first.")

        if isinstance(query_vars, str):
            query_vars = [query_vars]

        if evidence is None:
            evidence = {}

        result = self.inference.query(variables=query_vars, evidence=evidence)
        return result

    def average_causal_effect(self, treatment_var, treatment_values, outcome_var):
        """Calculate average causal effect between two values of a treatment variable"""
        val1, val2 = treatment_values
        p1 = self.do_intervention(treatment_var, val1, outcome_var)
        p2 = self.do_intervention(treatment_var, val2, outcome_var)
        
        if self.variable_mappings[outcome_var]["type"] == "binary":
            if hasattr(p1, "values") and hasattr(p2, "values"):
                ace = p2.values[1] - p1.values[1]
            else:
                ace = p2.get_value(**{outcome_var: 1}) - p1.get_value(**{outcome_var: 1})
        else:
            if hasattr(p1, "values") and hasattr(p2, "values"):
                exp1 = sum(val * prob for val, prob in zip(p1.state_names[outcome_var], p1.values))
                exp2 = sum(val * prob for val, prob in zip(p2.state_names[outcome_var], p2.values))
                ace = exp2 - exp1
            else:
                exp1 = sum(val * prob for val, prob in zip(p1.state_names[outcome_var], p1.values))
                exp2 = sum(val * prob for val, prob in zip(p2.state_names[outcome_var], p2.values))
                ace = exp2 - exp1
        
        return ace

    def do_intervention(self, intervention_var, intervention_value, query_var, samples=1000):
        """Perform causal intervention using do-calculus"""
        if self.inference is None:
            raise ValueError("Bayesian Network not built. Call build_bayesian_network() first.")

        intervened_model = BayesianNetwork(self.directed_edges)
        intervened_model.fit(self.df_discrete, estimator=MaximumLikelihoodEstimator)
        original_cpd = intervened_model.get_cpds(intervention_var)

        cardinality = original_cpd.cardinality[0]
        state_names = original_cpd.state_names[intervention_var]

        if intervention_value in state_names:
            intervention_index = state_names.index(intervention_value)
        else:
            intervention_index = min(
                range(len(state_names)),
                key=lambda i: abs(state_names[i] - intervention_value),
            )

        parents = list(self.graph.predecessors(intervention_var))

        if len(parents) == 0:
            new_values = np.zeros(cardinality)
            new_values[intervention_index] = 1.0
            new_values = new_values.reshape(-1, 1)
            new_cpd = TabularCPD(
                variable=intervention_var,
                variable_card=cardinality,
                values=new_values,
                state_names=original_cpd.state_names,
            )
        else:
            parent_combinations = int(
                np.prod([intervened_model.get_cpds(parent).cardinality[0] for parent in parents])
            )
            new_values = np.zeros((cardinality, parent_combinations))
            new_values[intervention_index, :] = 1.0

            evidence_card = [
                intervened_model.get_cpds(parent).cardinality[0] for parent in parents
            ]

            new_cpd = TabularCPD(
                variable=intervention_var,
                variable_card=cardinality,
                values=new_values,
                evidence=parents,
                evidence_card=evidence_card,
                state_names=original_cpd.state_names,
            )

        intervened_model.remove_cpds(intervention_var)
        intervened_model.add_cpds(new_cpd)

        try:
            intervened_inference = VariableElimination(intervened_model)
            result = intervened_inference.query(variables=[query_var])
            return result
        except Exception as e:
            return self._monte_carlo_intervention(
                intervention_var, intervention_value, query_var, samples
            )

    def _monte_carlo_intervention(self, intervention_var, intervention_value, query_var, samples=1000):
        """Monte Carlo sampling for intervention when exact inference fails"""
        topo_order = list(nx.topological_sort(self.graph))
        samples_list = []

        for sample_idx in range(samples):
            sample = {}
            for node in topo_order:
                if node == intervention_var:
                    sample[node] = intervention_value
                else:
                    parents = list(self.graph.predecessors(node))
                    if not parents:
                        cpd = self.model.get_cpds(node)
                        probs = cpd.values.flatten()
                        states = cpd.state_names[node]
                        sample[node] = np.random.choice(states, p=probs)
                    else:
                        parent_evidence = {p: sample[p] for p in parents}
                        try:
                            conditional_dist = self.inference.query(
                                variables=[node], evidence=parent_evidence
                            )
                            probs = conditional_dist.values
                            states = conditional_dist.state_names[node]
                            sample[node] = np.random.choice(states, p=probs)
                        except:
                            cpd = self.model.get_cpds(node)
                            states = cpd.state_names[node]
                            sample[node] = np.random.choice(states)

            samples_list.append(sample)

        query_values = [sample[query_var] for sample in samples_list]
        unique_values, counts = np.unique(query_values, return_counts=True)
        probabilities = counts / len(query_values)

        class MonteCarloResult:
            def __init__(self, variable, values, probs):
                self.variable = variable
                self.state_names = {variable: values}
                self.values = probs

        return MonteCarloResult(query_var, unique_values, probabilities)

    def rank_control_variables_by_ace(self, control_vars, outcome_vars=None, method='weighted_ace'):
        """Rank control variables by their Average Causal Effect (ACE) on outcome variables WITH CACHING"""
        if outcome_vars is None:
            outcome_vars = ["Collision", "Relaxed_Task_Result", "Global_Path_Score", "Local_Path_Score"]
        
        cache_key = (tuple(sorted(control_vars)), tuple(sorted(outcome_vars)), method)
        
        if cache_key in self._ace_ranking_cache:
            print(f"Using cached ACE rankings (skipping expensive computation)")
            return self._ace_ranking_cache[cache_key]
        
        print(f"Computing ACE rankings (this will be cached for future use)...")
        print(f"Ranking {len(control_vars)} control variables by ACE...")
        print(f"Outcome variables: {outcome_vars}")
        print(f"Ranking method: {method}")
        
        objective_weights = {
            "Relaxed_Task_Result": 5,
        }
        
        variable_scores = {}
        detailed_aces = {}
        
        for var in control_vars:
            if var not in self.variable_mappings:
                print(f"Warning: Variable {var} not found in mappings, skipping")
                continue
                
            vals = self.variable_mappings[var]["values"]
            print(f"\nAnalyzing {var} (values: {vals})...")
            
            var_aces = {}
            all_aces = []
            
            for outcome in outcome_vars:
                if outcome not in self.df.columns:
                    continue
                    
                outcome_aces = []
                pair_details = {}
                
                for val1, val2 in combinations(vals, 2):
                    try:
                        ace = self.average_causal_effect(var, (val1, val2), outcome)
                        outcome_aces.append(ace)
                        pair_details[f"{val1}→{val2}"] = ace
                        all_aces.append(abs(ace))
                        
                    except Exception as e:
                        print(f"Error calculating ACE for {var}: {val1}→{val2} on {outcome}: {e}")
                        continue
                
                if outcome_aces:
                    var_aces[outcome] = {
                        'max_abs_ace': max(outcome_aces, key=abs),
                        'mean_abs_ace': np.mean([abs(ace) for ace in outcome_aces]),
                        'details': pair_details
                    }
            
            detailed_aces[var] = var_aces
            
            weighted_score = 0
            total_weight = 0
            for outcome, ace_info in var_aces.items():
                weight = objective_weights.get(outcome, 1)
                weighted_score += weight * abs(ace_info['max_abs_ace'])
                total_weight += weight
            variable_scores[var] = weighted_score / total_weight if total_weight > 0 else 0
        
        ranked_variables = sorted(variable_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*60}")
        print("VARIABLE RANKING RESULTS")
        print(f"{'='*60}")
        
        for i, (var, score) in enumerate(ranked_variables):
            print(f"{i+1:2d}. {var:20s}: {score:.4f}")
            if var in detailed_aces:
                for outcome, ace_info in detailed_aces[var].items():
                    print(f"    {outcome:15s}: max_ACE={ace_info['max_abs_ace']:7.4f}, "
                        f"mean_ACE={ace_info['mean_abs_ace']:7.4f}")
        
        ranking_results = {
            'ranked_variables': [var for var, _ in ranked_variables],
            'variable_scores': variable_scores,
            'detailed_aces': detailed_aces,
            'ranking_method': method
        }
        
        self._ace_ranking_cache[cache_key] = ranking_results
        
        return ranking_results
    
    def precompute_ace_rankings(self, control_vars=None, outcome_vars=None):
        """Pre-compute ACE rankings to avoid delays during optimization"""
        if control_vars is None:
            control_vars = ["Controller", "Global_Planner", "Footprint_Type", 
                        "Inflation_Radius", "Cost_Scaling_Factor"]
        
        if outcome_vars is None:
            outcome_vars = ["Collision", "Relaxed_Task_Result", 
                        "Global_Path_Score", "Local_Path_Score"]
        
        print("\n" + "="*60)
        print("PRE-COMPUTING ACE RANKINGS (one-time cost)")
        print("="*60)
        
        start_time = time.time()
        self.rank_control_variables_by_ace(control_vars, outcome_vars)
        
        elapsed = time.time() - start_time
        print(f"\nACE ranking pre-computation completed in {elapsed:.2f}s")
        print("Future optimizations will use cached rankings (instant)")
        print("="*60)
    
    def find_optimal_configuration_hybrid(self, given_conditions, control_vars=None, objectives=None):
        """Find optimal configuration using hybrid causal approach"""
        if control_vars is None:
            control_vars = ["Controller", "Global_Planner", "Footprint_Type", "Inflation_Radius", "Cost_Scaling_Factor"]
            
        if objectives is None:
            objectives = [
                {"variable": "Collision", "direction": "minimize", "weight": 4},
                {"variable": "Relaxed_Task_Result", "direction": "maximize", "weight": 5},
                {"variable": "Global_Path_Score", "direction": "maximize", "weight": 2},
                {"variable": "Local_Path_Score", "direction": "maximize", "weight": 3},
            ]
        
        available_control_vars = [var for var in control_vars if var not in given_conditions]
        print("Available control variables for optimization:", available_control_vars)
        
        if not available_control_vars:
            raise ValueError("No control variables available for optimization")
            
        print(f"Starting hybrid causal optimization...")
        print(f"Available control variables: {available_control_vars}")
        
        ranking_results = self.rank_control_variables_by_ace(
            available_control_vars, 
            outcome_vars=[obj["variable"] for obj in objectives],
            method='weighted_ace'
        )
        
        ranked_vars = ranking_results['ranked_variables']
        start_time = time.time()
        
        result = self._hybrid_search(given_conditions, ranked_vars, objectives)
        
        total_time = time.time() - start_time
        
        result.update({
            'ranking_results': ranking_results,
            'optimization_time': total_time,
            'variables_considered': ranked_vars,
        })
        
        print(f"\n{'='*60}")
        print("HYBRID CAUSAL OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Variables ranked by importance: {ranked_vars}")
        print(f"Optimization time: {total_time:.2f}s")
        
        if result['best_configuration']:
            best_config = result['best_configuration']
            config_str = ", ".join([f"{k}={v}" for k, v in best_config.items() 
                                   if k in ranked_vars])
            print(f"Best configuration: {config_str}")
            print(f"Best score: {best_config['Total_Score']:.4f}")
            print(f"Evaluations performed: {result.get('evaluations_performed', 'Unknown')}")
        
        return result

    def _hybrid_search(self, given_conditions, ranked_vars, objectives):
        """Hybrid approach: beam search for top variables, greedy for rest"""
        print(f"\n--- Hybrid Search ---")
        
        n_beam_vars = min(3, len(ranked_vars))
        beam_vars = ranked_vars[:n_beam_vars]
        greedy_vars = ranked_vars[n_beam_vars:]
        
        print(f"Beam search variables: {beam_vars}")
        print(f"Greedy search variables: {greedy_vars}")
        
        total_evaluations = 0
        
        if beam_vars:
            beam_result = self._beam_search(given_conditions, beam_vars, objectives, beam_width=2)
            best_beam_config = beam_result['best_configuration']
            total_evaluations += beam_result['evaluations_performed']
            
            intermediate_evidence = given_conditions.copy()
            for var in beam_vars:
                if var in best_beam_config:
                    intermediate_evidence[var] = best_beam_config[var]
        else:
            intermediate_evidence = given_conditions.copy()
            best_beam_config = {}
        
        if greedy_vars:
            greedy_result = self._greedy_sequential_search(intermediate_evidence, greedy_vars, objectives)
            total_evaluations += greedy_result['evaluations_performed']
            
            final_config = best_beam_config.copy()
            final_config.update(greedy_result['best_configuration'])
        else:
            final_config = best_beam_config
        
        final_evidence = given_conditions.copy()
        final_evidence.update({k: v for k, v in final_config.items() if k != 'Total_Score'})
        final_score = self._evaluate_configuration(final_evidence, objectives)
        final_config['Total_Score'] = final_score
        
        return {
            'best_configuration': final_config,
            'evaluations_performed': total_evaluations,
            'method_details': 'hybrid_beam_greedy',
            'beam_variables': beam_vars,
            'greedy_variables': greedy_vars
        }

    def _beam_search(self, given_conditions, ranked_vars, objectives, beam_width=2):
        """Beam search: maintain top-k candidates at each step"""
        print(f"  Beam Search (width={beam_width})")
        
        beam = [{'config': {}, 'evidence': given_conditions.copy(), 'score': 0}]
        evaluations = 0
        
        for i, var in enumerate(ranked_vars):
            print(f"    Step {i+1}: Expanding {var}")
            
            new_candidates = []
            var_values = self.variable_mappings[var]["values"]
            
            for candidate in beam:
                for value in var_values:
                    new_config = candidate['config'].copy()
                    new_config[var] = value
                    
                    new_evidence = candidate['evidence'].copy()
                    new_evidence[var] = value
                    
                    score = self._evaluate_configuration(new_evidence, objectives)
                    evaluations += 1
                    
                    new_candidates.append({
                        'config': new_config,
                        'evidence': new_evidence,
                        'score': score
                    })
            
            new_candidates.sort(key=lambda x: x['score'], reverse=True)
            beam = new_candidates[:beam_width]
        
        best_candidate = beam[0]
        final_config = {**best_candidate['config'], 'Total_Score': best_candidate['score']}
        
        return {
            'best_configuration': final_config,
            'final_beam': beam,
            'evaluations_performed': evaluations,
            'method_details': f'beam_search_width_{beam_width}'
        }

    def _greedy_sequential_search(self, given_conditions, ranked_vars, objectives):
        """Sequential greedy search"""
        current_config = {}
        current_evidence = given_conditions.copy()
        optimization_path = []
        evaluations = 0
        
        print(f"  Greedy Sequential Search")
        
        for i, var in enumerate(ranked_vars):
            print(f"    Step {i+1}: Optimizing {var}")
            
            var_values = self.variable_mappings[var]["values"]
            best_score = float('-inf')
            best_value = None
            
            for value in var_values:
                test_evidence = current_evidence.copy()
                test_evidence[var] = value
                
                score = self._evaluate_configuration(test_evidence, objectives)
                evaluations += 1
                
                if score > best_score:
                    best_score = score
                    best_value = value
            
            if best_value is not None:
                current_config[var] = best_value
                current_evidence[var] = best_value
                optimization_path.append({
                    'variable': var,
                    'selected_value': best_value,
                    'score': best_score
                })
        
        final_score = self._evaluate_configuration(current_evidence, objectives)
        final_config = {**current_config, 'Total_Score': final_score}
        
        return {
            'best_configuration': final_config,
            'optimization_path': optimization_path,
            'evaluations_performed': evaluations,
            'method_details': 'greedy_sequential'
        }

    def find_optimal_configuration_random_forest(self, given_conditions, control_vars=None, objectives=None, n_estimators=150):
        """Find optimal configuration using Random Forest Regressor"""
        if control_vars is None:
            control_vars = ["Controller", "Global_Planner", "Footprint_Type", "Inflation_Radius", "Cost_Scaling_Factor"]
            
        if objectives is None:
            objectives = [
                {"variable": "Collision", "direction": "minimize", "weight": 4},
                {"variable": "Relaxed_Task_Result", "direction": "maximize", "weight": 5},
                {"variable": "Global_Path_Score", "direction": "maximize", "weight": 2},
                {"variable": "Local_Path_Score", "direction": "maximize", "weight": 3},
            ]
        
        available_control_vars = [var for var in control_vars if var not in given_conditions]
        
        if not available_control_vars:
            raise ValueError("No control variables available for optimization")
            
        print(f"Starting Random Forest optimization...")
        print(f"Available control variables: {available_control_vars}")
        print(f"N_estimators: {n_estimators}")
        
        start_time = time.time()
        
        X_train, y_train = self._prepare_rf_training_data(given_conditions, available_control_vars, objectives)
        
        if X_train.empty:
            print("No valid training data available")
            return {'best_configuration': None, 'evaluations_performed': 0}
        
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        candidates = self._generate_rf_candidates(given_conditions, available_control_vars)
        
        X_candidates = []
        candidate_configs = []
        
        for config in candidates:
            features = []
            full_evidence = given_conditions.copy()
            full_evidence.update(config)
            
            for var in available_control_vars:
                features.append(full_evidence.get(var, 0))
                
            for var in given_conditions.keys():
                if var not in available_control_vars:
                    features.append(given_conditions[var])
            
            X_candidates.append(features)
            candidate_configs.append(config)
        
        X_candidates = pd.DataFrame(X_candidates, columns=X_train.columns)
        
        predicted_scores = rf_model.predict(X_candidates)
        
        best_idx = np.argmax(predicted_scores)
        best_config = candidate_configs[best_idx]
        best_predicted_score = predicted_scores[best_idx]
        
        full_evidence = given_conditions.copy()
        full_evidence.update(best_config)
        actual_score = self._evaluate_configuration(full_evidence, objectives)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("RANDOM FOREST OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Candidates evaluated: {len(candidates)}")
        print(f"Optimization time: {total_time:.2f}s")
        
        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        print(f"Best configuration: {config_str}")
        print(f"Predicted score: {best_predicted_score:.4f}")
        print(f"Actual score: {actual_score:.4f}")
        
        feature_importance = dict(zip(X_train.columns, rf_model.feature_importances_))
        print(f"Top 3 important features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        final_config = {**best_config, 'Total_Score': actual_score}
        
        return {
            'best_configuration': final_config,
            'predicted_score': best_predicted_score,
            'actual_score': actual_score,
            'rf_model': rf_model,
            'feature_importance': feature_importance,
            'optimization_time': total_time,
            'training_samples': len(X_train),
            'candidates_evaluated': len(candidates),
            'evaluations_performed': len(candidates),  # Add this for consistency with other methods
            'method_details': f'random_forest_n{n_estimators}'
        }

    # =====================================================================
    # COMPARING METHODS: Bayesian Optimization and CMA-ES
    # =====================================================================
    
    def find_optimal_configuration_bayesian_optimization(self, given_conditions, control_vars=None, 
                                                         objectives=None, n_calls=50, n_random_starts=10):
        """
        Find optimal configuration using Bayesian Optimization (scikit-optimize)
        
        Parameters:
        -----------
        given_conditions : dict
            Fixed conditions
        control_vars : list
            Variables to optimize
        objectives : list
            Objective functions
        n_calls : int
            Number of function evaluations
        n_random_starts : int
            Number of random initialization points
        """
        if control_vars is None:
            control_vars = ["Controller", "Global_Planner", "Footprint_Type", "Inflation_Radius", "Cost_Scaling_Factor"]
            
        if objectives is None:
            objectives = [
                {"variable": "Collision", "direction": "minimize", "weight": 4},
                {"variable": "Relaxed_Task_Result", "direction": "maximize", "weight": 5},
                {"variable": "Global_Path_Score", "direction": "maximize", "weight": 2},
                {"variable": "Local_Path_Score", "direction": "maximize", "weight": 3},
            ]
        
        available_control_vars = [var for var in control_vars if var not in given_conditions]
        
        if not available_control_vars:
            raise ValueError("No control variables available for optimization")
        
        print(f"Starting Bayesian Optimization...")
        print(f"Available control variables: {available_control_vars}")
        print(f"n_calls: {n_calls}, n_random_starts: {n_random_starts}")
        
        start_time = time.time()
        
     
        search_space = []
        var_names = []
        
        for var in available_control_vars:
            if var not in self.variable_mappings:
                continue
                
            values = self.variable_mappings[var]["values"]
            var_names.append(var)
            
            # Create categorical dimension for each variable
            search_space.append(Categorical(values, name=var))
        
        # Store evaluation history
        eval_history = []
        
        
        @use_named_args(search_space)
        def objective(**params):
            # Build evidence
            evidence = given_conditions.copy()
            evidence.update(params)
            
            # Evaluate configuration
            score = self._evaluate_configuration(evidence, objectives)
            
    
            eval_history.append({
                'params': params.copy(),
                'score': score
            })
            
         
            return -score
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            search_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=42,
            verbose=False
        )
        
        total_time = time.time() - start_time
        
   
        best_params = dict(zip(var_names, result.x))
        best_score = -result.fun  # Negate back to original score
        
        print(f"\n{'='*60}")
        print("BAYESIAN OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Optimization time: {total_time:.2f}s")
        print(f"Function evaluations: {len(result.func_vals)}")
        
        config_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
        print(f"Best configuration: {config_str}")
        print(f"Best score: {best_score:.4f}")
        
        final_config = {**best_params, 'Total_Score': best_score}
        
        return {
            'best_configuration': final_config,
            'optimization_time': total_time,
            'evaluations_performed': len(result.func_vals),
            'eval_history': eval_history,
            'bo_result': result,
            'method_details': f'bayesian_opt_n{n_calls}'
        }
    
    def find_optimal_configuration_cmaes(self, given_conditions, control_vars=None, 
                                         objectives=None, max_evaluations=200, sigma0=0.3):
        """
        Find optimal configuration using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
        
        Parameters:
        -----------
        given_conditions : dict
            Fixed conditions
        control_vars : list
            Variables to optimize
        objectives : list
            Objective functions
        max_evaluations : int
            Maximum number of function evaluations
        sigma0 : float
            Initial standard deviation (step size)
        """
        if control_vars is None:
            control_vars = ["Controller", "Global_Planner", "Footprint_Type", "Inflation_Radius", "Cost_Scaling_Factor"]
            
        if objectives is None:
            objectives = [
                {"variable": "Collision", "direction": "minimize", "weight": 4},
                {"variable": "Relaxed_Task_Result", "direction": "maximize", "weight": 5},
                {"variable": "Global_Path_Score", "direction": "maximize", "weight": 2},
                {"variable": "Local_Path_Score", "direction": "maximize", "weight": 3},
            ]
        
        available_control_vars = [var for var in control_vars if var not in given_conditions]
        
        if not available_control_vars:
            raise ValueError("No control variables available for optimization")
        
        print(f"Starting CMA-ES optimization...")
        print(f"Available control variables: {available_control_vars}")
        print(f"max_evaluations: {max_evaluations}, sigma0: {sigma0}")
        
        start_time = time.time()
        
   
        var_info = []
        for var in available_control_vars:
            if var in self.variable_mappings:
                values = self.variable_mappings[var]["values"]
                var_info.append({
                    'name': var,
                    'values': values,
                    'min_idx': 0,
                    'max_idx': len(values) - 1
                })
        
  
        def objective(x):
            """
            x: array of continuous values in [0, 1] for each variable
            """
            # Map continuous x to discrete variable values
            evidence = given_conditions.copy()
            
            for i, info in enumerate(var_info):
                # Map x[i] from [0, 1] to integer index
                idx = int(np.clip(x[i] * len(info['values']), 0, len(info['values']) - 1))
                evidence[info['name']] = info['values'][idx]
            
   
            score = self._evaluate_configuration(evidence, objectives)
            

            return -score
        
        # Initial solution: middle of the range
        x0 = np.ones(len(var_info)) * 0.5
        
        # Bounds: [0, 1] for each variable
        bounds = [[0, 1]] * len(var_info)
        

        opts = {
            'maxfevals': max_evaluations,
            'bounds': [0, 1],
            'verbose': -1,  # Silent mode
            'seed': 42
        }
        
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        eval_history = []
        
        while not es.stop():
            solutions = es.ask()
            fitness_values = [objective(x) for x in solutions]
            
            # Store evaluations
            for x, f in zip(solutions, fitness_values):
                eval_history.append({
                    'x': x.copy(),
                    'fitness': f
                })
            
            es.tell(solutions, fitness_values)
        
        total_time = time.time() - start_time
        
    
        best_x = es.result.xbest
        best_fitness = es.result.fbest
        best_score = -best_fitness
        
  
        best_config = {}
        for i, info in enumerate(var_info):
            idx = int(np.clip(best_x[i] * len(info['values']), 0, len(info['values']) - 1))
            best_config[info['name']] = info['values'][idx]
        
        print(f"\n{'='*60}")
        print("CMA-ES OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Optimization time: {total_time:.2f}s")
        print(f"Function evaluations: {es.result.evaluations}")
        
        config_str = ", ".join([f"{k}={v}" for k, v in best_config.items()])
        print(f"Best configuration: {config_str}")
        print(f"Best score: {best_score:.4f}")
        
        final_config = {**best_config, 'Total_Score': best_score}
        
        return {
            'best_configuration': final_config,
            'optimization_time': total_time,
            'evaluations_performed': es.result.evaluations,
            'eval_history': eval_history,
            'cma_result': es.result,
            'method_details': f'cmaes_maxeval{max_evaluations}'
        }
    

    def _prepare_rf_training_data(self, given_conditions, control_vars, objectives):
        """Prepare training data for Random Forest using EMPIRICAL PROBABILITIES from data"""
        
        filtered_df = self.df.copy().reset_index(drop=True)
        
        for var, value in given_conditions.items():
            if var in filtered_df.columns:
                if self.variable_mappings.get(var, {}).get("type") == "continuous":
                    tolerance = 0.01
                    mask = abs(filtered_df[var] - value) <= tolerance
                else:
                    mask = filtered_df[var] == value
                filtered_df = filtered_df[mask].reset_index(drop=True)
        
        if len(filtered_df) < 10:
            print(f"Too few samples ({len(filtered_df)}) with exact conditions, using full dataset")
            filtered_df = self.df.copy().reset_index(drop=True)
        
        feature_cols = control_vars + list(given_conditions.keys())
        feature_cols = [col for col in feature_cols if col in filtered_df.columns]
        feature_cols = list(dict.fromkeys(feature_cols))
        
        if not feature_cols:
            print("No valid feature columns found")
            return pd.DataFrame(), pd.Series()
        
        X = filtered_df[feature_cols].copy()
        
        y_scores = []
        for idx in range(len(filtered_df)):
            row = filtered_df.iloc[idx]
            config = {var: row[var] for var in control_vars if var in row.index}
            
            score = self._evaluate_configuration_probabilistic_from_data(
                config, objectives, filtered_df, control_vars
            )
            y_scores.append(score)
        
        y = pd.Series(y_scores, index=X.index)
        
        valid_mask = ~(y.isna() | y.isin([float('inf'), float('-inf')]))
        valid_indices = valid_mask[valid_mask].index
        
        X = X.loc[valid_indices].reset_index(drop=True)
        y = y.loc[valid_indices].reset_index(drop=True)
        
        print(f"Training data prepared: {len(X)} samples with {len(feature_cols)} features")
        print(f"Using EMPIRICAL PROBABILITIES for fair comparison with causal method")
        
        return X, y

    def _evaluate_configuration_probabilistic_from_data(self, config, objectives, data_subset, control_vars):
        """Evaluate configuration using EMPIRICAL PROBABILITIES from data"""
        try:
            mask = pd.Series([True] * len(data_subset))
            for var, value in config.items():
                if var in data_subset.columns:
                    if self.variable_mappings.get(var, {}).get("type")== "continuous":
                        tolerance = 0.01
                        mask &= (abs(data_subset[var] - value) <= tolerance)
                    else:
                        mask &= (data_subset[var] == value)
            
            matching_rows = data_subset[mask]
            
            if len(matching_rows) < 5:
                best_match_count = 0
                best_matches = matching_rows
                
                for skip_var in config.keys():
                    temp_mask = pd.Series([True] * len(data_subset))
                    for var, value in config.items():
                        if var == skip_var:
                            continue
                        if var in data_subset.columns:
                            if self.variable_mappings.get(var, {}).get("type") == "continuous":
                                tolerance = 0.01
                                temp_mask &= (abs(data_subset[var] - value) <= tolerance)
                            else:
                                temp_mask &= (data_subset[var] == value)
                    
                    temp_matches = data_subset[temp_mask]
                    if len(temp_matches) > best_match_count:
                        best_match_count = len(temp_matches)
                        best_matches = temp_matches
                
                matching_rows = best_matches
            
            if len(matching_rows) < 3:
                matching_rows = data_subset
            
            total_score = 0
            
            for objective in objectives:
                obj_var = objective["variable"]
                direction = objective["direction"]
                weight = objective["weight"]
                
                if obj_var not in matching_rows.columns:
                    continue
                
                if self.variable_mappings[obj_var]["type"] == "binary":
                    values = matching_rows[obj_var].values
                    prob_1 = np.mean(values)
                    prob_0 = 1 - prob_1
                    
                    if direction == "maximize":
                        score = prob_1
                    else:
                        score = prob_0
                        
                else:
                    expected_value = np.mean(matching_rows[obj_var].values)
                    score = expected_value if direction == "maximize" else -expected_value
                
                total_score += weight * score
            
            return total_score
            
        except Exception as e:
            print(f"Error in probabilistic evaluation: {e}")
            return 0
    
    def _generate_rf_candidates(self, given_conditions, control_vars, max_candidates=1000):
        """Generate candidate configurations for Random Forest evaluation"""
        
        variable_values = {}
        for var in control_vars:
            if var in self.variable_mappings:
                variable_values[var] = self.variable_mappings[var]["values"]
        
        if not variable_values:
            return []
        
        var_names = list(variable_values.keys())
        value_combinations = list(product(*[variable_values[var] for var in var_names]))
        
        if len(value_combinations) > max_candidates:
            indices = np.random.choice(len(value_combinations), max_candidates, replace=False)
            value_combinations = [value_combinations[i] for i in indices]
        
        candidates = []
        for combination in value_combinations:
            config = dict(zip(var_names, combination))
            candidates.append(config)
        
        return candidates
    
    def _evaluate_configuration(self, evidence, objectives):
        """OPTIMIZED evaluation with batch queries and caching"""
        try:
            interventions = {
                var: val for var, val in evidence.items()
                if var in self.variable_mappings
            }
            
            if not interventions:
                return float('-inf')
            
            obj_vars = [obj["variable"] for obj in objectives 
                    if obj["variable"] in self.df.columns]
            
            if not obj_vars:
                return float('-inf')
            
            results = self._query_interventions_cached(interventions, obj_vars)
            
            total_score = 0.0
            
            for objective in objectives:
                obj_var = objective["variable"]
                direction = objective["direction"]
                weight = objective["weight"]
                
                if obj_var not in results:
                    continue
                
                result = results[obj_var]
                
                if self.variable_mappings[obj_var]["type"] == "binary":
                    states = list(result.state_names[obj_var])
                    probs = result.values
                    
                    if direction == "maximize":
                        if 1 in states:
                            idx = states.index(1)
                            score = probs[idx]
                        else:
                            idx = int(np.argmax(states))
                            score = probs[idx]
                    else:
                        if 0 in states:
                            idx = states.index(0)
                            score = probs[idx]
                        else:
                            idx = int(np.argmin(states))
                            score = probs[idx]
                else:
                    states = list(result.state_names[obj_var])
                    probs = result.values
                    expected_val = float(np.sum(
                        np.array(states, dtype=float) * np.array(probs, dtype=float)
                    ))
                    score = expected_val if direction == "maximize" else -expected_val
                
                total_score += weight * float(score)
            
            return float(total_score)
            
        except Exception as e:
            print(f"Error in _evaluate_configuration: {e}")
            return float('-inf')

    def _query_interventions_cached(self, interventions: dict, query_vars: list):
        """Query multiple variables from cached intervened model"""
        cache_key = tuple(sorted((k, v) for k, v in interventions.items()))
        
        if cache_key in self._intervention_cache:
            self._cache_hits += 1
            intervened_model = self._intervention_cache[cache_key]
        else:
            self._cache_misses += 1
            intervened_model = self._build_intervened_model(interventions)
            
            if len(self._intervention_cache) < 1000:
                self._intervention_cache[cache_key] = intervened_model
            else:
                if len(self._intervention_cache) >= 1000:
                    keys_to_remove = list(self._intervention_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self._intervention_cache[key]
                self._intervention_cache[cache_key] = intervened_model
        
        try:
            intervened_inference = VariableElimination(intervened_model)
            results = {}
            
            for query_var in query_vars:
                try:
                    result = intervened_inference.query(variables=[query_var])
                    results[query_var] = result
                except Exception as e:
                    print(f"Exact inference failed for {query_var}, using MC sampling")
                    results[query_var] = self._monte_carlo_single_var(
                        intervened_model, query_var, samples=500
                    )
            
            return results
            
        except Exception as e:
            print(f"Batch query failed: {e}, falling back to Monte Carlo")
            return self._monte_carlo_multi_vars(interventions, query_vars, samples=500)

    def _build_intervened_model(self, interventions: dict):
        """Build intervened Bayesian Network with do-interventions applied"""
        if self.inference is None:
            raise ValueError("Bayesian Network not built. Call build_bayesian_network() first.")
        
        intervened_model = BayesianNetwork(self.directed_edges)
        intervened_model.fit(self.df_discrete, estimator=MaximumLikelihoodEstimator)
        
        for intervention_var, intervention_value in interventions.items():
            if intervention_var not in [cpd.variable for cpd in intervened_model.get_cpds()]:
                continue
            
            original_cpd = intervened_model.get_cpds(intervention_var)
            cardinality = original_cpd.cardinality[0]
            state_names = original_cpd.state_names[intervention_var]
            
            if intervention_value in state_names:
                intervention_index = state_names.index(intervention_value)
            else:
                intervention_index = min(
                    range(len(state_names)),
                    key=lambda i: abs(state_names[i] - intervention_value) 
                    if isinstance(state_names[i], (int, float)) else float('inf')
                )
            
            parents = list(self.graph.predecessors(intervention_var))
            
            if len(parents) == 0:
                new_values = np.zeros(cardinality)
                new_values[intervention_index] = 1.0
                new_values = new_values.reshape(-1, 1)
                new_cpd = TabularCPD(
                    variable=intervention_var,
                    variable_card=cardinality,
                    values=new_values,
                    state_names=original_cpd.state_names,
                )
            else:
                parent_combinations = int(
                    np.prod([intervened_model.get_cpds(parent).cardinality[0] 
                            for parent in parents])
                )
                new_values = np.zeros((cardinality, parent_combinations))
                new_values[intervention_index, :] = 1.0
                
                evidence_card = [
                    intervened_model.get_cpds(parent).cardinality[0] 
                    for parent in parents
                ]
                
                new_cpd = TabularCPD(
                    variable=intervention_var,
                    variable_card=cardinality,
                    values=new_values,
                    evidence=parents,
                    evidence_card=evidence_card,
                    state_names=original_cpd.state_names,
                )
            
            intervened_model.remove_cpds(intervention_var)
            intervened_model.add_cpds(new_cpd)
        
        return intervened_model
    
    def _monte_carlo_single_var(self, intervened_model, query_var, samples=500):
        """Monte Carlo sampling for single variable when exact inference fails"""
        topo_order = list(nx.topological_sort(self.graph))
        samples_list = []
        
        try:
            temp_inference = VariableElimination(intervened_model)
        except:
            temp_inference = None
        
        for _ in range(samples):
            sample = {}
            for node in topo_order:
                cpd = intervened_model.get_cpds(node)
                
                if np.max(cpd.values) == 1.0 and np.sum(cpd.values > 0.99) == 1:
                    fixed_state_idx = np.argmax(cpd.values.flatten())
                    sample[node] = cpd.state_names[node][fixed_state_idx]
                else:
                    parents = list(self.graph.predecessors(node))
                    if not parents:
                        probs = cpd.values.flatten()
                        states = cpd.state_names[node]
                        sample[node] = np.random.choice(states, p=probs)
                    else:
                        parent_evidence = {p: sample[p] for p in parents if p in sample}
                        try:
                            if temp_inference:
                                conditional_dist = temp_inference.query(
                                    variables=[node], evidence=parent_evidence
                                )
                                probs = conditional_dist.values
                                states = conditional_dist.state_names[node]
                                sample[node] = np.random.choice(states, p=probs)
                            else:
                                raise Exception("No inference available")
                        except:
                            states = cpd.state_names[node]
                            sample[node] = np.random.choice(states)
            
            samples_list.append(sample)
        
        query_values = [s[query_var] for s in samples_list]
        unique_values, counts = np.unique(query_values, return_counts=True)
        probabilities = counts / len(query_values)
        
        class MonteCarloResult:
            def __init__(self, variable, values, probs):
                self.variable = variable
                self.state_names = {variable: list(values)}
                self.values = probs
        
        return MonteCarloResult(query_var, unique_values, probabilities)
    
    def _monte_carlo_multi_vars(self, interventions: dict, query_vars: list, samples=500):
        """Monte Carlo sampling for multiple variables when exact inference fails"""
        topo_order = list(nx.topological_sort(self.graph))
        samples_list = []
        
        for _ in range(samples):
            sample = {}
            for node in topo_order:
                if node in interventions:
                    sample[node] = interventions[node]
                else:
                    parents = list(self.graph.predecessors(node))
                    if not parents:
                        cpd = self.model.get_cpds(node)
                        probs = cpd.values.flatten()
                        states = cpd.state_names[node]
                        sample[node] = np.random.choice(states, p=probs)
                    else:
                        parent_evidence = {p: sample[p] for p in parents if p in sample}
                        try:
                            conditional_dist = self.inference.query(
                                variables=[node], evidence=parent_evidence
                            )
                            probs = conditional_dist.values
                            states = conditional_dist.state_names[node]
                            sample[node] = np.random.choice(states, p=probs)
                        except:
                            cpd = self.model.get_cpds(node)
                            states = cpd.state_names[node]
                            sample[node] = np.random.choice(states)
            
            samples_list.append(sample)
        
        results = {}
        for query_var in query_vars:
            query_values = [s[query_var] for s in samples_list]
            unique_values, counts = np.unique(query_values, return_counts=True)
            probabilities = counts / len(query_values)
            
            class MonteCarloResult:
                def __init__(self, variable, values, probs):
                    self.variable = variable
                    self.state_names = {variable: list(values)}
                    self.values = probs
            
            results[query_var] = MonteCarloResult(query_var, unique_values, probabilities)
        
        return results

    def print_cache_stats(self):
        """Print cache performance statistics"""
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = (self._cache_hits / total) * 100
            print(f"\n{'='*60}")
            print("CACHE STATISTICS")
            print(f"{'='*60}")
            print(f"Cache hits: {self._cache_hits}")
            print(f"Cache misses: {self._cache_misses}")
            print(f"Hit rate: {hit_rate:.1f}%")
            print(f"Cache size: {len(self._intervention_cache)} models")
            print(f"{'='*60}")

    def clear_cache(self):
        """Clear the intervention cache"""
        self._intervention_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        print("Cache cleared")
    
    def compare_optimization_methods(self, test_conditions_list, control_vars=None, 
                                    objectives=None, n_trials=3, 
                                    include_bayesian=True, include_cmaes=True,
                                    bo_n_calls=50, cmaes_max_evals=200):
        """
        Compare ALL optimization methods: Hybrid Causal, Random Forest, Bayesian Optimization, CMA-ES
        
        Parameters:
        -----------
        test_conditions_list : list of dict
            List of test conditions to evaluate
        control_vars : list, optional
            Control variables to optimize
        objectives : list, optional
            Objective functions
        n_trials : int
            Number of trials for each method
        include_bayesian : bool
            Whether to include Bayesian Optimization
        include_cmaes : bool
            Whether to include CMA-ES
        bo_n_calls : int
            Number of calls for Bayesian Optimization
        cmaes_max_evals : int
            Maximum evaluations for CMA-ES
        """
        if control_vars is None:
            control_vars = ["Controller", "Global_Planner", "Footprint_Type", "Inflation_Radius", "Cost_Scaling_Factor"]
            
        if objectives is None:
            objectives = [
                {"variable": "Collision", "direction": "minimize", "weight": 4},
                {"variable": "Relaxed_Task_Result", "direction": "maximize", "weight": 5},
                {"variable": "Global_Path_Score", "direction": "maximize", "weight": 2},
                {"variable": "Local_Path_Score", "direction": "maximize", "weight": 3},
            ]

        comparison_results = {
            'method_performance': {},
            'detailed_results': {},
            'summary': {},
            'control_variables': control_vars
        }

        print("="*80)
        print("COMPREHENSIVE OPTIMIZATION METHOD COMPARISON")
        print("="*80)
        print(f"Test conditions: {len(test_conditions_list)}")
        print(f"Trials per method: {n_trials}")
        
        # Define which methods to compare
        methods = ['hybrid_causal', 'random_forest']
        if include_bayesian:
            methods.append('bayesian_optimization')
        if include_cmaes:
            methods.append('cmaes')
        
        print(f"Methods to compare: {methods}")
        
        for method_name in methods:
            print(f"\n{'='*60}")
            print(f"Evaluating {method_name.upper().replace('_', ' ')} method...")
            print(f"{'='*60}")
            
            method_scores = []
            method_times = []
            method_evaluations = []
            detailed_results = []

            for trial in range(n_trials):
                print(f"\nTrial {trial + 1}/{n_trials}")
                
                trial_scores = []
                trial_times = []
                trial_evaluations = []

                for i, conditions in enumerate(test_conditions_list):
                    print(f"  Condition set {i+1}/{len(test_conditions_list)}")
                    start_time = time.time()
                    
                    try:
                        if method_name == 'hybrid_causal':
                            result = self.find_optimal_configuration_hybrid(
                                conditions, control_vars, objectives
                            )
                        elif method_name == 'random_forest':
                            result = self.find_optimal_configuration_random_forest(
                                conditions, control_vars, objectives, n_estimators=150
                            )
                        elif method_name == 'bayesian_optimization':
                            result = self.find_optimal_configuration_bayesian_optimization(
                                conditions, control_vars, objectives, n_calls=bo_n_calls
                            )
                        elif method_name == 'cmaes':
                            result = self.find_optimal_configuration_cmaes(
                                conditions, control_vars, objectives, max_evaluations=cmaes_max_evals
                            )
                        
                        end_time = time.time()
                        
                        if result and result['best_configuration']:
                            score = result['best_configuration']['Total_Score']
                            evaluations = result.get('evaluations_performed', 0)
                        else:
                            score = float('-inf')
                            evaluations = 0
                        
                        trial_scores.append(score)
                        trial_times.append(end_time - start_time)
                        trial_evaluations.append(evaluations)

                        detailed_results.append({
                            'trial': trial,
                            'condition_set': i,
                            'conditions': conditions,
                            'score': score,
                            'time': end_time - start_time,
                            'evaluations': evaluations,
                            'configuration': result['best_configuration'] if result else None
                        })

                    except Exception as e:
                        print(f"    Error in {method_name} trial {trial}, condition {i}: {e}")
                        trial_scores.append(float('-inf'))
                        trial_times.append(float('inf'))
                        trial_evaluations.append(0)

                method_scores.extend(trial_scores)
                method_times.extend(trial_times)
                method_evaluations.extend(trial_evaluations)
                
                # Clear cache between trials for causal method
                if method_name == 'hybrid_causal':
                    self.print_cache_stats()
                    self.clear_cache()

            # Calculate statistics
            valid_scores = [s for s in method_scores if s != float('-inf')]
            valid_times = [t for t in method_times if t != float('inf')]
            valid_evaluations = [e for e in method_evaluations if e > 0]
            
            if valid_scores:
                comparison_results['method_performance'][method_name] = {
                    'mean_score': np.mean(valid_scores),
                    'std_score': np.std(valid_scores),
                    'min_score': np.min(valid_scores),
                    'max_score': np.max(valid_scores),
                    'mean_time': np.mean(valid_times) if valid_times else float('inf'),
                    'std_time': np.std(valid_times) if valid_times else 0,
                    'mean_evaluations': np.mean(valid_evaluations) if valid_evaluations else 0,
                    'success_rate': len(valid_scores) / len(method_scores),
                    'efficiency': np.mean(valid_scores) / np.mean(valid_evaluations) if valid_evaluations else 0
                }
                
                comparison_results['detailed_results'][method_name] = detailed_results

        # Calculate comparison metrics (all vs all)
        method_names = list(comparison_results['method_performance'].keys())
        
        if len(method_names) >= 2:
            # Find best method for each metric
            best_score_method = max(method_names, 
                key=lambda m: comparison_results['method_performance'][m]['mean_score'])
            best_time_method = min(method_names, 
                key=lambda m: comparison_results['method_performance'][m]['mean_time'])
            best_efficiency_method = max(method_names, 
                key=lambda m: comparison_results['method_performance'][m]['efficiency'])
            
            comparison_results['summary'] = {
                'winner_by_score': best_score_method,
                'winner_by_time': best_time_method,
                'winner_by_efficiency': best_efficiency_method,
                'methods_compared': method_names
            }

        # Print comparison summary
        print("\n" + "="*80)
        print("METHOD COMPARISON SUMMARY")
        print("="*80)
        
        for method, performance in comparison_results['method_performance'].items():
            method_display = method.replace('_', ' ').title()
            print(f"\n{method_display}:")
            print(f"  Mean Score: {performance['mean_score']:.4f} ± {performance['std_score']:.4f}")
            print(f"  Score Range: [{performance['min_score']:.4f}, {performance['max_score']:.4f}]")
            print(f"  Mean Time: {performance['mean_time']:.4f}s ± {performance['std_time']:.4f}s")
            print(f"  Mean Evaluations: {performance['mean_evaluations']:.0f}")
            print(f"  Efficiency (Score/Eval): {performance['efficiency']:.6f}")
            print(f"  Success Rate: {performance['success_rate']:.2%}")

        if 'summary' in comparison_results:
            summary = comparison_results['summary']
            print(f"\nCOMPARISON WINNERS:")
            print(f"  Best Score: {summary['winner_by_score'].replace('_', ' ').title()}")
            print(f"  Fastest: {summary['winner_by_time'].replace('_', ' ').title()}")
            print(f"  Most Efficient: {summary['winner_by_efficiency'].replace('_', ' ').title()}")

        return comparison_results

    def plot_method_comparison(self, comparison_results, save_path=None, also_png=False, 
                               monochrome=False, decimals=dict(score=3, time=2, evals=0, eff=4)):
        """
        Publication-quality plotting function with 4 charts in a single row
        Includes: Performance, Computation Time, Search Space Exploration, Efficiency
        """
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        # Publication-quality style - NO GRID
        mpl.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'axes.linewidth': 1.2,
            'figure.dpi': 150,
            'savefig.dpi': 600,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': False,  # DISABLE GRID
            'grid.alpha': 0.3,
            'grid.linestyle': ':',
        })

        # Colorblind-friendly color scheme (using Wong palette)
        if monochrome:
            bar_colors = ['0.2', '0.4', '0.6', '0.8']
            hatches = ['', '///', 'xxx', '\\\\\\']
        else:
            # Wong colorblind-friendly palette
            bar_colors = [
                '#0072B2',  # Blue - Hybrid Causal
                '#D55E00',  # Vermillion/Orange - Random Forest
                '#009E73',  # Bluish green - Bayesian Opt
                '#CC79A7'   # Reddish purple - CMA-ES
            ]
            hatches = ['', '', '', '']

        methods = list(comparison_results['method_performance'].keys())
        
        print(f"\n=== DEBUG: Methods found in results ===")
        print(f"Methods: {methods}")
        for m in methods:
            print(f"\n{m}:")
            print(f"  mean_evaluations: {comparison_results['method_performance'][m]['mean_evaluations']}")
            print(f"  efficiency: {comparison_results['method_performance'][m]['efficiency']}")
        
        # Create method names for publication
        method_names = []
        for m in methods:
            if m == 'hybrid_causal':
                method_names.append('Hybrid Causal')
            elif m == 'random_forest':
                method_names.append('Random Forest')
            elif m == 'bayesian_optimization':
                method_names.append('Bayesian Opt.')
            elif m == 'cmaes':
                method_names.append('CMA-ES')
            else:
                method_names.append(m.replace('_', ' ').title())

        def get_metric(mkey):
            values = []
            for m in methods:
                val = comparison_results['method_performance'][m][mkey]
                # Handle potential inf values
                if val == float('inf'):
                    val = 0  # Replace inf with 0 for visualization
                values.append(val)
            return values

        mean_scores = get_metric('mean_score')
        std_scores  = get_metric('std_score')
        mean_times  = get_metric('mean_time')
        mean_evals  = get_metric('mean_evaluations')
        efficiencies= get_metric('efficiency')
        
        print(f"\n=== DEBUG: Extracted metrics ===")
        print(f"mean_evals: {mean_evals}")
        print(f"efficiencies: {efficiencies}")

        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.15, wspace=0.3)

        x_pos = np.arange(len(method_names))
        bar_width = 0.65

 
        def annotate_bars_clean(ax, bars, values, decimals_int, suffix=""):
            ylim = ax.get_ylim()
            span = ylim[1] - ylim[0]
            for b, v in zip(bars, values):
                if v != float('-inf') and v != float('inf') and v > 0:
                    y_pos = b.get_height() + (0.03 * span)
                    ax.text(b.get_x() + b.get_width()/2, y_pos,
                            f"{v:.{decimals_int}f}{suffix}",
                            ha='center', va='bottom',
                            fontsize=10, fontweight='normal',
                            color='black')

        # 1) PERFORMANCE
        ax = axes[0]
        bars = ax.bar(x_pos, mean_scores, yerr=std_scores, 
                     capsize=4, width=bar_width,
                     color=bar_colors[:len(method_names)], 
                     edgecolor='black', linewidth=1.0, alpha=0.85,
                     error_kw={'linewidth': 1.5, 'ecolor': 'black'})
        for b, h in zip(bars, hatches[:len(method_names)]): 
            b.set_hatch(h)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.set_ylabel('Mean Score', fontweight='bold')
        ax.set_title('(a) Performance', fontweight='bold', loc='left', pad=12)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.15)
        annotate_bars_clean(ax, bars, mean_scores, decimals['score'])
        ax.set_axisbelow(True)

        # 2) COMPUTATION TIME
        ax = axes[1]
        bars = ax.bar(x_pos, mean_times, width=bar_width,
                     color=bar_colors[:len(method_names)], 
                     edgecolor='black', linewidth=1.0, alpha=0.85)
        for b, h in zip(bars, hatches[:len(method_names)]): 
            b.set_hatch(h)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.set_ylabel('Time (seconds)', fontweight='bold')
        ax.set_title('(b) Computation Time', fontweight='bold', loc='left', pad=12)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.15)
        annotate_bars_clean(ax, bars, mean_times, decimals['time'], suffix="s")
        ax.set_axisbelow(True)

        # 3) EVALUATIONS (log scale) - FIXED to show all methods with labels ON TOP of bars
        ax = axes[2]
        
        # Filter out zero/invalid values for log scale but keep track of positions
        plot_evals = []
        valid_indices = []
        for i, v in enumerate(mean_evals):
            if v > 0:
                plot_evals.append(v)
                valid_indices.append(i)
        
        if not plot_evals:  # If all values are 0, use linear scale
            print("WARNING: All evaluation counts are 0, using linear scale")
            bars = ax.bar(x_pos, mean_evals, width=bar_width,
                         color=bar_colors[:len(method_names)], 
                         edgecolor='black', linewidth=1.0, alpha=0.85)
        else:
            # Plot all bars including zeros
            bars = ax.bar(x_pos, mean_evals, width=bar_width,
                         color=bar_colors[:len(method_names)], 
                         edgecolor='black', linewidth=1.0, alpha=0.85)
            ax.set_yscale('log')
            ax.set_ylim(bottom=max(0.1, min([v for v in mean_evals if v > 0]) * 0.5))
            
        for b, h in zip(bars, hatches[:len(method_names)]): 
            b.set_hatch(h)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.set_ylabel('Number of Evaluations', fontweight='bold')
        ax.set_title('(c) Search Space Exploration', fontweight='bold', loc='left', pad=12)
        
        if plot_evals:  # Only format log scale if we have valid data
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(axis='y', style='plain')
        
        # Annotations EXACTLY ON TOP of ALL bars
        for i, (b, v) in enumerate(zip(bars, mean_evals)):
            if v > 0:
                # Place label exactly at the top of the bar
                y_pos = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, y_pos,
                        f"{int(v)}",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        color='black')
        
        ax.set_axisbelow(True)

        # 4) EFFICIENCY (Score per Evaluation) - FIXED to show all methods
        ax = axes[3]
        bars = ax.bar(x_pos, efficiencies, width=bar_width,
                     color=bar_colors[:len(method_names)], 
                     edgecolor='black', linewidth=1.0, alpha=0.85)
        for b, h in zip(bars, hatches[:len(method_names)]): 
            b.set_hatch(h)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.set_ylabel('Score / Evaluation', fontweight='bold')
        ax.set_title('(d) Efficiency', fontweight='bold', loc='left', pad=12)
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.15)
        annotate_bars_clean(ax, bars, efficiencies, decimals['eff'])
        ax.set_axisbelow(True)

        # Overall title
        if 'summary' in comparison_results:
            winner_key = comparison_results['summary'].get('winner_by_score', 'unknown')
            winner_display = {
                'hybrid_causal': 'Hybrid Causal',
                'random_forest': 'Random Forest',
                'bayesian_optimization': 'Bayesian Optimization',
                'cmaes': 'CMA-ES'
            }.get(winner_key, winner_key.replace('_', ' ').title())
            
            fig.suptitle(f'Comparison of Optimization Methods',
                        fontsize=15, fontweight='bold', y=0.98)

        # Save with high quality
        if save_path:
            base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            
            # Save PDF (vector format for publications)
            plt.savefig(f"{base}_comparison.pdf", 
                       format='pdf', bbox_inches='tight', 
                       dpi=600, pad_inches=0.2)
            print(f"✓ Saved: {base}_comparison.pdf")
            
            # Optionally save PNG
            if also_png:
                plt.savefig(f"{base}_comparison.png", 
                           format='png', bbox_inches='tight',
                           dpi=600, pad_inches=0.2)
                print(f"✓ Saved: {base}_comparison.png")

        plt.show()


def main():
    # Load data
    data = pd.read_csv("/home/forough/Desktop/causal_navigation/src/causal inference/results_dataset.csv")
    
    # Define causal structure
    directed_edges = [
        ("Collision", "Relaxed_Task_Result"),
        ("Collision","Local_Path_Score"),
        ("Controller", "Relaxed_Task_Result"),
        ("Controller", "Collision"),
        ("Controller", "Local_Path_Score"),
        ("Controller", "Min_Local_Dist_To_Obstacl"),
        ("Global_Planner", "Collision"),
        ("Global_Planner", "Relaxed_Task_Result"),
        ("Global_Planner", "Global_Path_Score"),
        ("Global_Planner", "Min_Global_Dist_To_Obst"),
        ("Min_Global_Dist_To_Obst", "Global_Path_Score"),
        ("Inflation_Radius", "Min_Local_Dist_To_Obstacl"),
        ("Inflation_Radius", "Min_Global_Dist_To_Obst"),
        ("Inflation_Radius", "Global_Path_Score"),
        ("Inflation_Radius", "Local_Path_Score"),
        ("Footprint_Type", "Min_Local_Dist_To_Obstacl"),
        ("Footprint_Type", "Collision"),
        ("Footprint_Type", "Min_Global_Dist_To_Obst"),
        ("Min_Local_Dist_To_Obstacl", "Local_Path_Score"),
        ("Min_Local_Dist_To_Obstacl", "Collision"),
        ("Cost_Scaling_Factor", "Local_Path_Score"),  
        ("Cost_Scaling_Factor", "Global_Path_Score"),
    ]
    
    # Initialize model
    enhanced_model = EnhancedCausalModel(data, directed_edges)
    enhanced_model.build_bayesian_network()
    
    # Pre-compute ACE rankings
    control_variables = ["Controller", "Global_Planner", "Footprint_Type", 
                        "Inflation_Radius", "Cost_Scaling_Factor"]
    enhanced_model.precompute_ace_rankings(control_variables)
    
    # Test conditions
    test_conditions = [
        {"Cost_Scaling_Factor": 25},
    ]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON: ALL METHODS")
    print("="*80)
    print("Methods: Causal (Hybrid), Random Forest, Bayesian Optimization, CMA-ES")
    
    # Run comprehensive comparison
    comparison_results = enhanced_model.compare_optimization_methods(
        test_conditions, 
        control_variables,
        n_trials=2,
        include_bayesian=True,
        include_cmaes=True,
        bo_n_calls=50,  # Adjust as needed
        cmaes_max_evals=200  # Adjust as needed
    )
    
    # Plot results
    enhanced_model.plot_method_comparison(
        comparison_results, 
        save_path="/home/forough/Desktop/causal_navigation/src/causal inference/evaluation_results/"
    )
    
    print("\n" + "="*80)
    print("FINAL CONCLUSIONS")
    print("="*80)
    
    if 'summary' in comparison_results:
        summary = comparison_results['summary']
        print("Based on this comprehensive comparison:")
        print(f"• Best Score: {summary['winner_by_score'].replace('_', ' ').title()}")
        print(f"• Fastest: {summary['winner_by_time'].replace('_', ' ').title()}")
        print(f"• Most Efficient: {summary['winner_by_efficiency'].replace('_', ' ').title()}")
        
        print(f"\nAll methods compared: {', '.join([m.replace('_', ' ').title() for m in summary['methods_compared']])}")


if __name__ == "__main__":
    main()