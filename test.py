import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import itertools
import warnings
warnings.filterwarnings("ignore")

# Import your existing TouristRecommendationSystem class
from satisfaction_estimator import TouristRecommendationSystem

class MultiStrategyEvaluator:
    """
    A framework to evaluate multiple recommendation strategies against each other.
    """
    def __init__(self, dataset_path, train_size=0.3, random_state=42):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path: Path to the tourism dataset CSV
            train_size: Proportion of data to use for training the models
            random_state: Random state for reproducibility
        """
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.random_state = random_state
        
        # Dictionary to store recommendation strategies
        self.strategies = {}
        
        # Load the dataset
        self.df = pd.read_csv(dataset_path)
        
        # Split into train and test sets
        self.train_df, self.test_df = train_test_split(
            self.df, train_size=train_size, random_state=random_state
        )
        
        print(f"Dataset loaded: {len(self.df)} total records")
        print(f"Train set: {len(self.train_df)} records")
        print(f"Test set: {len(self.test_df)} records")
        
        # Extract all unique interests and sites from the dataset
        self.all_interests = self._extract_unique_interests()
        self.all_sites = self._extract_unique_sites()
        
        print(f"Unique interests found: {len(self.all_interests)}")
        print(f"Unique sites found: {len(self.all_sites)}")
    
    def _extract_unique_interests(self):
        """Extract all unique interests from the dataset."""
        all_interests = set()
        for interests in self.df['Interests']:
            if isinstance(interests, str):
                try:
                    if interests.startswith('[') and interests.endswith(']'):
                        interests = eval(interests)
                    else:
                        interests = [i.strip() for i in interests.split(',')]
                except:
                    interests = [interests]
            all_interests.update(interests)
        return list(all_interests)
    
    def _extract_unique_sites(self):
        """Extract all unique sites from the dataset."""
        all_sites = set()
        for sites in self.df['Sites Visited']:
            if isinstance(sites, str):
                try:
                    if sites.startswith('[') and sites.endswith(']'):
                        sites = eval(sites)
                    else:
                        sites = [s.strip() for s in sites.split(',')]
                except:
                    sites = [sites]
            all_sites.update(sites)
        return list(all_sites)
    
    def add_strategy(self, name, recommender_class, **kwargs):
        """
        Add a recommendation strategy to be evaluated.
        
        Args:
            name: Name of the strategy
            recommender_class: Class that implements the recommendation strategy
            **kwargs: Additional arguments to pass to the recommender constructor
        """
        print(f"Adding strategy: {name}")
        strategy = recommender_class(self.train_df, **kwargs)
        self.strategies[name] = strategy
        print(f"Strategy '{name}' added successfully.")
        return strategy
    
    def generate_simulated_tourists(self, num_tourists, random_seed=None):
        """
        Generate simulated tourists based on patterns in the test dataset.
        
        Args:
            num_tourists: Number of tourists to simulate
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        simulated_tourists = []
        
        # Generate simulated tourists
        for _ in range(num_tourists):
            # Sample age following distribution of real data
            age = np.random.choice(self.df['Age'])
            
            # Sample random interests (1-4 interests)
            num_interests = np.random.randint(1, 5)
            interests = random.sample(self.all_interests, min(num_interests, len(self.all_interests)))
            
            # Sample preferred duration
            preferred_duration = np.random.choice(self.df['Preferred Tour Duration'])
            
            # Sample accessibility
            accessibility = bool(np.random.choice([True, False]))
            
            simulated_tourists.append({
                'age': age,
                'interests': interests,
                'preferred_duration': preferred_duration,
                'accessibility': accessibility
            })
        
        return simulated_tourists
    
    def evaluate_strategies_on_test_set(self, random_seed=None):
        """
        Evaluate all registered recommendation strategies using the actual test dataset.
        
        Args:
            random_seed: Random seed for reproducibility
        
        Returns:
            Dictionary containing evaluation results for all strategies
        """
        if not self.strategies:
            raise ValueError("No strategies registered. Use add_strategy() to add strategies.")
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        print(f"Evaluating {len(self.strategies)} recommendation strategies on test set ({len(self.test_df)} records)...")
        
        # Results storage for each strategy
        results = {name: [] for name in self.strategies}
        recommendations = {name: [] for name in self.strategies}
        actual_satisfaction = []
        
        # Process each tourist in the test set
        for _, tourist in tqdm(self.test_df.iterrows()):
            # Extract tourist attributes
            age = tourist['Age']
            
            # Parse interests
            if isinstance(tourist['Interests'], str):
                if tourist['Interests'].startswith('[') and tourist['Interests'].endswith(']'):
                    interests = eval(tourist['Interests'])
                else:
                    interests = [i.strip() for i in tourist['Interests'].split(',')]
            else:
                interests = tourist['Interests']
            
            # Get preferred duration and accessibility
            preferred_duration = tourist['Preferred Tour Duration']
            accessibility = tourist['Accessibility']
            
            # Get actual sites visited and satisfaction
            if isinstance(tourist['Sites Visited'], str):
                if tourist['Sites Visited'].startswith('[') and tourist['Sites Visited'].endswith(']'):
                    actual_sites = eval(tourist['Sites Visited'])
                else:
                    actual_sites = [s.strip() for s in tourist['Sites Visited'].split(',')]
            else:
                actual_sites = tourist['Sites Visited']
            
            actual_satisfaction.append(tourist['Satisfaction'])
            
            # Get recommendations from each strategy
            for name, strategy in self.strategies.items():
                recommendation = strategy.recommend_sites(
                    age, interests, preferred_duration, accessibility
                )
                
                # Save the recommendation
                recommendations[name].append(recommendation)
                
                # Use the predicted satisfaction directly
                results[name].append(recommendation['expected_satisfaction'])
        
        # Calculate statistics for each strategy
        statistics = {}
        for name in self.strategies:
            avg_satisfaction = np.mean(results[name])
            std_satisfaction = np.std(results[name])
            
            # Calculate error metrics
            mae = np.mean(np.abs(np.array(results[name]) - np.array(actual_satisfaction)))
            rmse = np.sqrt(np.mean((np.array(results[name]) - np.array(actual_satisfaction))**2))
            
            statistics[name] = {
                'satisfaction_scores': results[name],
                'recommendations': recommendations[name],
                'avg_satisfaction': avg_satisfaction,
                'std_satisfaction': std_satisfaction,
                'min_satisfaction': min(results[name]),
                'max_satisfaction': max(results[name]),
                'median_satisfaction': np.median(results[name]),
                'mae': mae,
                'rmse': rmse
            }
        
        # Perform pairwise statistical tests
        pairwise_tests = {}
        for name1, name2 in itertools.combinations(self.strategies.keys(), 2):
            t_stat, p_value = stats.ttest_ind(results[name1], results[name2])
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((statistics[name1]['std_satisfaction']**2 + 
                                  statistics[name2]['std_satisfaction']**2) / 2)
            effect_size = abs(statistics[name1]['avg_satisfaction'] - 
                             statistics[name2]['avg_satisfaction']) / pooled_std
            
            pairwise_tests[f"{name1} vs {name2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05,
                'better': name1 if statistics[name1]['avg_satisfaction'] > statistics[name2]['avg_satisfaction'] else name2
            }
        
        # Find the best strategy
        strategy_names = list(self.strategies.keys())
        avg_satisfactions = [statistics[name]['avg_satisfaction'] for name in strategy_names]
        best_strategy = strategy_names[np.argmax(avg_satisfactions)]
        
        # Package results
        evaluation_results = {
            'statistics': statistics,
            'pairwise_tests': pairwise_tests,
            'best_strategy': best_strategy,
            'num_tourists': len(self.test_df),
            'actual_satisfaction': actual_satisfaction
        }
        
        return evaluation_results
    
    def evaluate_strategies(self, num_tourists=200, noise_level=0.2, random_seed=None):
        """
        Evaluate all registered recommendation strategies using simulated tourists.
        
        Args:
            num_tourists: Number of tourists to simulate
            noise_level: Standard deviation of random noise to add to simulated satisfaction
            random_seed: Random seed for reproducibility
        
        Returns:
            Dictionary containing evaluation results for all strategies
        """
        if not self.strategies:
            raise ValueError("No strategies registered. Use add_strategy() to add strategies.")
        
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        print(f"Evaluating {len(self.strategies)} recommendation strategies...")
        
        # Generate simulated tourists
        simulated_tourists = self.generate_simulated_tourists(num_tourists, random_seed)
        
        # Results storage for each strategy
        results = {name: [] for name in self.strategies}
        recommendations = {name: [] for name in self.strategies}
        
        # Process each tourist with each strategy
        for i, tourist in enumerate(tqdm(simulated_tourists)):
            for name, strategy in self.strategies.items():
                # Get recommendations
                recommendation = strategy.recommend_sites(
                    tourist['age'],
                    tourist['interests'],
                    tourist['preferred_duration'],
                    tourist['accessibility']
                )
                
                # Save the recommendation
                recommendations[name].append(recommendation)
                
                # Simulate tourist's actual satisfaction
                satisfaction = recommendation['expected_satisfaction']
                satisfaction += np.random.normal(0, noise_level)
                satisfaction = max(0, min(5, satisfaction))  # Clamp to 0-5 range
                
                results[name].append(satisfaction)
        
        # Calculate statistics for each strategy
        statistics = {}
        for name in self.strategies:
            avg_satisfaction = np.mean(results[name])
            std_satisfaction = np.std(results[name])
            
            statistics[name] = {
                'satisfaction_scores': results[name],
                'recommendations': recommendations[name],
                'avg_satisfaction': avg_satisfaction,
                'std_satisfaction': std_satisfaction,
                'min_satisfaction': min(results[name]),
                'max_satisfaction': max(results[name]),
                'median_satisfaction': np.median(results[name])
            }
        
        # Perform pairwise statistical tests
        pairwise_tests = {}
        for name1, name2 in itertools.combinations(self.strategies.keys(), 2):
            t_stat, p_value = stats.ttest_ind(results[name1], results[name2])
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((statistics[name1]['std_satisfaction']**2 + 
                                  statistics[name2]['std_satisfaction']**2) / 2)
            effect_size = abs(statistics[name1]['avg_satisfaction'] - 
                             statistics[name2]['avg_satisfaction']) / pooled_std
            
            pairwise_tests[f"{name1} vs {name2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05,
                'better': name1 if statistics[name1]['avg_satisfaction'] > statistics[name2]['avg_satisfaction'] else name2
            }
        
        # Find the best strategy
        strategy_names = list(self.strategies.keys())
        avg_satisfactions = [statistics[name]['avg_satisfaction'] for name in strategy_names]
        best_strategy = strategy_names[np.argmax(avg_satisfactions)]
        
        # Package results
        evaluation_results = {
            'statistics': statistics,
            'pairwise_tests': pairwise_tests,
            'best_strategy': best_strategy,
            'num_tourists': num_tourists
        }
        
        return evaluation_results
    
    def visualize_results(self, aggregated_results):
        """
        Visualize the aggregated evaluation results across multiple runs.
        
        Args:
            aggregated_results: Dictionary containing aggregated results across multiple runs
        """
        avg_statistics = aggregated_results['avg_statistics']
        std_statistics = aggregated_results['std_statistics']
        avg_pairwise_tests = aggregated_results['avg_pairwise_tests']
        best_strategy_counts = aggregated_results['best_strategy_counts']
        num_runs = aggregated_results['num_runs']
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 20))
        
        # 1. Average satisfaction comparison with error bars
        plt.subplot(4, 2, 1)
        strategy_names = list(avg_statistics.keys())
        avg_satisfactions = [avg_statistics[name]['avg_satisfaction'] for name in strategy_names]
        std_across_runs = [std_statistics[name]['avg_satisfaction'] for name in strategy_names]
        
        bars = plt.bar(strategy_names, avg_satisfactions, yerr=std_across_runs, 
                       capsize=10, alpha=0.7)
        
        # Highlight the most frequent best strategy
        best_strategy = max(best_strategy_counts.items(), key=lambda x: x[1])[0]
        best_idx = strategy_names.index(best_strategy)
        bars[best_idx].set_color('green')
        
        plt.ylabel('Average Satisfaction')
        plt.title(f'Average Satisfaction by Strategy (Across {num_runs} Runs)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(avg_satisfactions):
            plt.text(i, v + 0.1, f'{v:.2f}±{std_across_runs[i]:.2f}', ha='center')
        
        # 2. Error metrics - MAE
        plt.subplot(4, 2, 2)
        avg_maes = [avg_statistics[name]['avg_mae'] for name in strategy_names]
        std_maes = [std_statistics[name]['mae'] for name in strategy_names]
        
        bars = plt.bar(strategy_names, avg_maes, yerr=std_maes, 
                       capsize=10, alpha=0.7)
        
        plt.ylabel('Mean Absolute Error')
        plt.title('Mean Absolute Error by Strategy (Lower is Better)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(avg_maes):
            plt.text(i, v + 0.05, f'{v:.2f}±{std_maes[i]:.2f}', ha='center')
        
        # 3. Error metrics - RMSE
        plt.subplot(4, 2, 3)
        avg_rmses = [avg_statistics[name]['avg_rmse'] for name in strategy_names]
        std_rmses = [std_statistics[name]['rmse'] for name in strategy_names]
        
        bars = plt.bar(strategy_names, avg_rmses, yerr=std_rmses, 
                       capsize=10, alpha=0.7)
        
        plt.ylabel('Root Mean Squared Error')
        plt.title('RMSE by Strategy (Lower is Better)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(avg_rmses):
            plt.text(i, v + 0.05, f'{v:.2f}±{std_rmses[i]:.2f}', ha='center')
        
        # 4. Consistency of the strategies (standard deviation across runs)
        plt.subplot(4, 2, 4)
        std_satisfactions = [std_statistics[name]['avg_satisfaction'] for name in strategy_names]
        
        bars = plt.bar(strategy_names, std_satisfactions, alpha=0.7)
        
        plt.ylabel('Standard Deviation')
        plt.title('Consistency of Strategies Across Runs')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(std_satisfactions):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # 5. Best strategy count
        plt.subplot(4, 2, 5)
        best_strategy_names = list(best_strategy_counts.keys())
        best_strategy_counts_values = [best_strategy_counts[name] for name in best_strategy_names]
        
        bars = plt.bar(best_strategy_names, best_strategy_counts_values, alpha=0.7)
        
        plt.ylabel('Number of Times Selected as Best')
        plt.title(f'Best Strategy Frequency (Out of {num_runs} Runs)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(best_strategy_counts_values):
            plt.text(i, v + 0.1, str(v), ha='center')
        
        # 6. Box plot showing distribution of average satisfaction across runs
        plt.subplot(4, 2, 6)
        data = []
        labels = []
        
        for name in strategy_names:
            if 'satisfaction_across_runs' in avg_statistics[name]:
                data.append(avg_statistics[name]['satisfaction_across_runs'])
                labels.append(name)
        
        plt.boxplot(data, labels=labels)
        plt.ylabel('Average Satisfaction')
        plt.title('Distribution of Average Satisfaction Across Runs')
        plt.xticks(rotation=45, ha='right')
        
        # 7. Pairwise statistical significance heatmap
        plt.subplot(4, 2, 7)
        
        # Create matrix of p-values
        n_strategies = len(strategy_names)
        p_value_matrix = np.ones((n_strategies, n_strategies))
        
        for test_name, test_result in avg_pairwise_tests.items():
            name1, name2 = test_name.split(' vs ')
            idx1 = strategy_names.index(name1)
            idx2 = strategy_names.index(name2)
            p_value_matrix[idx1, idx2] = test_result['p_value']
            p_value_matrix[idx2, idx1] = test_result['p_value']
        
        # Set diagonal to 1 (not significant)
        np.fill_diagonal(p_value_matrix, 1)
        
        # Create heatmap
        sns.heatmap(p_value_matrix, annot=True, cmap='coolwarm_r', 
                    xticklabels=strategy_names, yticklabels=strategy_names, 
                    vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
        
        plt.title('Pairwise Statistical Significance (p-values)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 8. Pairwise comparison text summary
        plt.subplot(4, 2, 8)
        plt.axis('off')
        
        # Create a summary table of pairwise tests
        cell_text = []
        for test_name, test_result in avg_pairwise_tests.items():
            better = test_result['better']
            p_value = test_result['p_value']
            effect = test_result['effect_size']
            significant_count = test_result['significant_count']
            
            cell_text.append([
                test_name, 
                better, 
                f"{p_value:.4f}", 
                f"{effect:.2f}", 
                f"{significant_count}/{num_runs}"
            ])
        
        plt.table(cellText=cell_text,
                 colLabels=['Comparison', 'Better Strategy', 'Avg P-value', 'Avg Effect Size', 'Times Significant'],
                 loc='center',
                 cellLoc='center',
                 colWidths=[0.25, 0.2, 0.15, 0.15, 0.15])
        
        plt.title('Pairwise Statistical Tests Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig('multi_run_strategy_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary text report
        print(f"\n--- Strategy Evaluation Results (Across {num_runs} Runs) ---\n")
        
        # Most frequent best strategy
        most_common_best = max(best_strategy_counts.items(), key=lambda x: x[1])
        print(f"Most Frequent Best Strategy: {most_common_best[0]} (Selected {most_common_best[1]}/{num_runs} times)")
        
        print("\nRanking by Average Satisfaction:")
        
        # Sort strategies by average satisfaction
        sorted_strategies = sorted(strategy_names, 
                                  key=lambda x: avg_statistics[x]['avg_satisfaction'], 
                                  reverse=True)
        
        for i, name in enumerate(sorted_strategies):
            avg_sat = avg_statistics[name]['avg_satisfaction']
            std_sat = std_statistics[name]['avg_satisfaction']
            print(f"{i+1}. {name}: {avg_sat:.2f} ± {std_sat:.2f}")
        
        print("\nPairwise Comparisons:")
        for test_name, test_result in avg_pairwise_tests.items():
            significance = f"significant in {test_result['significant_count']}/{num_runs} runs"
            print(f"- {test_name}: {test_result['better']} is better on average (p={test_result['p_value']:.4f}, {significance})")


# Various recommendation strategies remain the same as in the original code
# ...

class PopularityRecommender:
    """
    Recommends the most popular sites overall.
    """
    def __init__(self, train_df, **kwargs):
        self.train_df = train_df
        
        # Calculate site popularity
        self.site_popularity = {}
        self.site_avg_satisfaction = {}
        self.site_durations = {}
        
        # Process all sites in the training data
        for _, row in train_df.iterrows():
            # Get sites
            if isinstance(row['Sites Visited'], str):
                if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                    sites = eval(row['Sites Visited'])
                else:
                    sites = [s.strip() for s in row['Sites Visited'].split(',')]
            else:
                sites = row['Sites Visited']
            
            # Update site popularity
            for site in sites:
                if site not in self.site_popularity:
                    self.site_popularity[site] = 0
                    self.site_avg_satisfaction[site] = []
                
                self.site_popularity[site] += 1
                self.site_avg_satisfaction[site].append(row['Satisfaction'])
            
            # Calculate average duration per site
            if len(sites) > 0:
                avg_duration = row['Tour Duration'] / len(sites)
                for site in sites:
                    if site not in self.site_durations:
                        self.site_durations[site] = []
                    self.site_durations[site].append(avg_duration)
        
        # Calculate average satisfaction for each site
        for site in self.site_avg_satisfaction:
            if self.site_avg_satisfaction[site]:
                self.site_avg_satisfaction[site] = sum(self.site_avg_satisfaction[site]) / len(self.site_avg_satisfaction[site])
            else:
                self.site_avg_satisfaction[site] = 0
        
        # Calculate average duration for each site
        for site in self.site_durations:
            if self.site_durations[site]:
                self.site_durations[site] = sum(self.site_durations[site]) / len(self.site_durations[site])
            else:
                self.site_durations[site] = 1.0  # Default to 1 hour if unknown
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        # Sort sites by popularity
        sorted_sites = sorted(self.site_popularity.keys(), 
                             key=lambda x: self.site_popularity[x], 
                             reverse=True)
        
        # Select top sites within duration constraint
        recommended_sites = []
        total_duration = 0
        
        for site in sorted_sites:
            site_duration = self.site_durations.get(site, 1.0)
            if total_duration + site_duration <= preferred_duration:
                recommended_sites.append(site)
                total_duration += site_duration
                
                if len(recommended_sites) >= top_k:
                    break
        
        # Calculate expected satisfaction
        if recommended_sites:
            expected_satisfaction = sum(self.site_avg_satisfaction.get(site, 0) for site in recommended_sites) / len(recommended_sites)
        else:
            expected_satisfaction = 0
        
        return {
            'recommended_sites': recommended_sites,
            'expected_satisfaction': expected_satisfaction,
            'total_duration': total_duration
        }


class InterestMatchingRecommender:
    """
    Recommends sites that best match the tourist's interests.
    """
    def __init__(self, train_df, **kwargs):
        self.train_df = train_df
        
        # Map interests to sites
        self.interest_to_sites = {}
        self.site_avg_satisfaction = {}
        self.site_durations = {}
        
        # Process all sites in the training data
        for _, row in train_df.iterrows():
            # Get sites
            if isinstance(row['Sites Visited'], str):
                if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                    sites = eval(row['Sites Visited'])
                else:
                    sites = [s.strip() for s in row['Sites Visited'].split(',')]
            else:
                sites = row['Sites Visited']
            
            # Get interests
            if isinstance(row['Interests'], str):
                if row['Interests'].startswith('[') and row['Interests'].endswith(']'):
                    interests = eval(row['Interests'])
                else:
                    interests = [i.strip() for i in row['Interests'].split(',')]
            else:
                interests = row['Interests']
            
            # Update site-interest mapping
            for site in sites:
                if site not in self.site_avg_satisfaction:
                    self.site_avg_satisfaction[site] = []
                
                self.site_avg_satisfaction[site].append(row['Satisfaction'])
                
                # Link site to interests
                for interest in interests:
                    if interest not in self.interest_to_sites:
                        self.interest_to_sites[interest] = {}
                    
                    if site not in self.interest_to_sites[interest]:
                        self.interest_to_sites[interest][site] = 0
                    
                    self.interest_to_sites[interest][site] += 1
            
            # Calculate average duration per site
            if len(sites) > 0:
                avg_duration = row['Tour Duration'] / len(sites)
                for site in sites:
                    if site not in self.site_durations:
                        self.site_durations[site] = []
                    self.site_durations[site].append(avg_duration)
        
        # Calculate average satisfaction for each site
        for site in self.site_avg_satisfaction:
            if self.site_avg_satisfaction[site]:
                self.site_avg_satisfaction[site] = sum(self.site_avg_satisfaction[site]) / len(self.site_avg_satisfaction[site])
            else:
                self.site_avg_satisfaction[site] = 0
        
        # Calculate average duration for each site
        for site in self.site_durations:
            if self.site_durations[site]:
                self.site_durations[site] = sum(self.site_durations[site]) / len(self.site_durations[site])
            else:
                self.site_durations[site] = 1.0  # Default to 1 hour if unknown
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        # Score sites based on interest matching
        site_scores = {}
        
        for interest in interests:
            if interest in self.interest_to_sites:
                for site, count in self.interest_to_sites[interest].items():
                    if site not in site_scores:
                        site_scores[site] = 0
                    site_scores[site] += count
        
        # If no matches, return empty recommendation
        if not site_scores:
            return {
                'recommended_sites': [],
                'expected_satisfaction': 0,
                'total_duration': 0
            }
        
        # Sort sites by interest matching score
        sorted_sites = sorted(site_scores.keys(), key=lambda x: site_scores[x], reverse=True)
        
        # Select top sites within duration constraint
        recommended_sites = []
        total_duration = 0
        
        for site in sorted_sites:
            site_duration = self.site_durations.get(site, 1.0)
            if total_duration + site_duration <= preferred_duration:
                recommended_sites.append(site)
                total_duration += site_duration
                
                if len(recommended_sites) >= top_k:
                    break
        
        # Calculate expected satisfaction
        if recommended_sites:
            expected_satisfaction = sum(self.site_avg_satisfaction.get(site, 0) for site in recommended_sites) / len(recommended_sites)
        else:
            expected_satisfaction = 0
        
        return {
            'recommended_sites': recommended_sites,
            'expected_satisfaction': expected_satisfaction,
            'total_duration': total_duration
        }

class LeastPopularRecommender:
    """
    Recommends the least popular (least visited) sites.
    This contrarian approach focuses on off-the-beaten-path experiences.
    """
    def __init__(self, train_df, **kwargs):
        self.train_df = train_df
        
        # Count site occurrences to determine popularity
        self.site_popularity = {}
        self.site_avg_satisfaction = {}
        self.site_durations = {}
        
        # Process all sites in the training data
        for _, row in train_df.iterrows():
            # Get sites
            if isinstance(row['Sites Visited'], str):
                if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                    sites = eval(row['Sites Visited'])
                else:
                    sites = [s.strip() for s in row['Sites Visited'].split(',')]
            else:
                sites = row['Sites Visited']
            
            # Update site popularity (count occurrences)
            for site in sites:
                if site not in self.site_popularity:
                    self.site_popularity[site] = 0
                    self.site_avg_satisfaction[sit00000000000000000e] = []
                
                self.site_popularity[site] += 1
                self.site_avg_satisfaction[site].append(row['Satisfaction'])
            
            # Calculate average duration per site
            if len(sites) > 0:
                avg_duration = row['Tour Duration'] / len(sites)
                for site in sites:
                    if site not in self.site_durations:
                        self.site_durations[site] = []
                    self.site_durations[site].append(avg_duration)
        
        # Calculate average satisfaction for each site
        for site in self.site_avg_satisfaction:
            if self.site_avg_satisfaction[site]:
                self.site_avg_satisfaction[site] = sum(self.site_avg_satisfaction[site]) / len(self.site_avg_satisfaction[site])
            else:
                self.site_avg_satisfaction[site] = 0
        
        # Calculate average duration for each site
        for site in self.site_durations:
            if self.site_durations[site]:
                self.site_durations[site] = sum(self.site_durations[site]) / len(self.site_durations[site])
            else:
                self.site_durations[site] = 1.0  # Default to 1 hour if unknown
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        # Sort sites by popularity (ascending order to get least popular first)
        sorted_sites = sorted(self.site_popularity.keys(), 
                             key=lambda x: self.site_popularity[x], 
                             reverse=False)
        
        # Select top least popular sites within duration constraint
        recommended_sites = []
        total_duration = 0
        
        for site in sorted_sites:
            site_duration = self.site_durations.get(site, 1.0)
            if total_duration + site_duration <= preferred_duration:
                recommended_sites.append(site)
                total_duration += site_duration
                
                if len(recommended_sites) >= top_k:
                    break
        
        # Calculate expected satisfaction
        if recommended_sites:
            expected_satisfaction = sum(self.site_avg_satisfaction.get(site, 0) for site in recommended_sites) / len(recommended_sites)
        else:
            expected_satisfaction = 0
        
        return {
            'recommended_sites': recommended_sites,
            'expected_satisfaction': expected_satisfaction,
            'total_duration': total_duration
        }


class RandomRecommender:
    """
    Recommends sites completely randomly.
    This approach serves as a baseline and may introduce serendipitous discoveries.
    """
    def __init__(self, train_df, **kwargs):
        self.train_df = train_df
        
        # Extract random seed if provided
        self.random_seed = kwargs.get('random_state', None)
        if self.random_seed is not None:
            import random
            random.seed(self.random_seed)
        
        # Keep track of all possible sites
        self.all_sites = set()
        self.site_avg_satisfaction = {}
        self.site_durations = {}
        
        # Process all sites in the training data
        for _, row in train_df.iterrows():
            # Get sites
            if isinstance(row['Sites Visited'], str):
                if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                    sites = eval(row['Sites Visited'])
                else:
                    sites = [s.strip() for s in row['Sites Visited'].split(',')]
            else:
                sites = row['Sites Visited']
            
            # Add to all sites collection
            for site in sites:
                self.all_sites.add(site)
                
                if site not in self.site_avg_satisfaction:
                    self.site_avg_satisfaction[site] = []
                
                self.site_avg_satisfaction[site].append(row['Satisfaction'])
            
            # Calculate average duration per site
            if len(sites) > 0:
                avg_duration = row['Tour Duration'] / len(sites)
                for site in sites:
                    if site not in self.site_durations:
                        self.site_durations[site] = []
                    self.site_durations[site].append(avg_duration)
        
        # Convert set to list for random selection
        self.all_sites = list(self.all_sites)
        
        # Calculate average satisfaction for each site
        for site in self.site_avg_satisfaction:
            if self.site_avg_satisfaction[site]:
                self.site_avg_satisfaction[site] = sum(self.site_avg_satisfaction[site]) / len(self.site_avg_satisfaction[site])
            else:
                self.site_avg_satisfaction[site] = 0
        
        # Calculate average duration for each site
        for site in self.site_durations:
            if self.site_durations[site]:
                self.site_durations[site] = sum(self.site_durations[site]) / len(self.site_durations[site])
            else:
                self.site_durations[site] = 1.0  # Default to 1 hour if unknown
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        import random
        
        # Shuffle all sites randomly
        all_sites_shuffled = self.all_sites.copy()
        random.shuffle(all_sites_shuffled)
        
        # Select sites within duration constraint
        recommended_sites = []
        total_duration = 0
        
        for site in all_sites_shuffled:
            site_duration = self.site_durations.get(site, 1.0)
            if total_duration + site_duration <= preferred_duration:
                recommended_sites.append(site)
                total_duration += site_duration
                
                if len(recommended_sites) >= top_k:
                    break
        
        # Calculate expected satisfaction
        if recommended_sites:
            expected_satisfaction = sum(self.site_avg_satisfaction.get(site, 0) for site in recommended_sites) / len(recommended_sites)
        else:
            expected_satisfaction = 0
        
        return {
            'recommended_sites': recommended_sites,
            'expected_satisfaction': expected_satisfaction,
            'total_duration': total_duration
        }

class SatisfactionMaximizingRecommender:
    """
    Recommends sites with the highest historical satisfaction ratings.
    """
    def __init__(self, train_df, **kwargs):
        self.train_df = train_df
        
        # Calculate site satisfaction
        self.site_avg_satisfaction = {}
        self.site_durations = {}
        
        # Process all sites in the training data
        for _, row in train_df.iterrows():
            # Get sites
            if isinstance(row['Sites Visited'], str):
                if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                    sites = eval(row['Sites Visited'])
                else:
                    sites = [s.strip() for s in row['Sites Visited'].split(',')]
            else:
                sites = row['Sites Visited']
            
            # Update site satisfaction
            for site in sites:
                if site not in self.site_avg_satisfaction:
                    self.site_avg_satisfaction[site] = []
                
                self.site_avg_satisfaction[site].append(row['Satisfaction'])
            
            # Calculate average duration per site
            if len(sites) > 0:
                avg_duration = row['Tour Duration'] / len(sites)
                for site in sites:
                    if site not in self.site_durations:
                        self.site_durations[site] = []
                    self.site_durations[site].append(avg_duration)
        
        # Calculate average satisfaction for each site
        for site in self.site_avg_satisfaction:
            if self.site_avg_satisfaction[site]:
                self.site_avg_satisfaction[site] = sum(self.site_avg_satisfaction[site]) / len(self.site_avg_satisfaction[site])
            else:
                self.site_avg_satisfaction[site] = 0
        
        # Calculate average duration for each site
        for site in self.site_durations:
            if self.site_durations[site]:
                self.site_durations[site] = sum(self.site_durations[site]) / len(self.site_durations[site])
            else:
                self.site_durations[site] = 1.0  # Default to 1 hour if unknown
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        # Sort sites by satisfaction rating
        sorted_sites = sorted(self.site_avg_satisfaction.keys(), 
                             key=lambda x: self.site_avg_satisfaction[x], 
                             reverse=True)
        
        # Select top sites within duration constraint
        recommended_sites = []
        total_duration = 0
        
        for site in sorted_sites:
            site_duration = self.site_durations.get(site, 1.0)
            if total_duration + site_duration <= preferred_duration:
                recommended_sites.append(site)
                total_duration += site_duration
                
                if len(recommended_sites) >= top_k:
                    break
        
        # Calculate expected satisfaction
        if recommended_sites:
            expected_satisfaction = sum(self.site_avg_satisfaction.get(site, 0) for site in recommended_sites) / len(recommended_sites)
        else:
            expected_satisfaction = 0
        
        return {
            'recommended_sites': recommended_sites,
            'expected_satisfaction': expected_satisfaction,
            'total_duration': total_duration
        }


class RandomForestRecommender:
    """
    Uses Random Forest to predict satisfaction for different site combinations.
    """
    def __init__(self, train_df, **kwargs):
        self.train_df = train_df
        
        # Extract features and targets
        X, y, feature_columns = self._preprocess_data(train_df)
        
        # Train random forest model
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=kwargs.get('random_state', 42))
        self.model.fit(X, y)
        print("Random Forest model trained.")
        
        # Store feature columns for prediction
        self.feature_columns = feature_columns
        
        # Extract all unique sites and calculate durations
        self.all_sites = set()
        self.site_durations = {}
        
        for _, row in train_df.iterrows():
            # Get sites
            if isinstance(row['Sites Visited'], str):
                if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                    sites = eval(row['Sites Visited'])
                else:
                    sites = [s.strip() for s in row['Sites Visited'].split(',')]
            else:
                sites = row['Sites Visited']
            
            self.all_sites.update(sites)
            
            # Calculate average duration per site
            if len(sites) > 0:
                avg_duration = row['Tour Duration'] / len(sites)
                for site in sites:
                    if site not in self.site_durations:
                        self.site_durations[site] = []
                    self.site_durations[site].append(avg_duration)
        
        # Calculate average duration for each site
        for site in self.site_durations:
            if self.site_durations[site]:
                self.site_durations[site] = sum(self.site_durations[site]) / len(self.site_durations[site])
            else:
                self.site_durations[site] = 1.0  # Default to 1 hour if unknown
        
        self.all_sites = list(self.all_sites)
        
        # Store scalers
        self.age_scaler = StandardScaler()
        self.age_scaler.fit(train_df[['Age']])
        
        self.duration_scaler = StandardScaler()
        self.duration_scaler.fit(train_df[['Preferred Tour Duration']])
    
    def _preprocess_data(self, df):
        """Preprocess the dataset for Random Forest."""
        df_clean = df.copy()
        
        # Handle categorical data and lists
        # 1. Process interests - create one-hot encoding
        all_interests = set()
        for interests in df_clean['Interests']:
            # Convert string representation to list if needed
            if isinstance(interests, str):
                try:
                    if interests.startswith('[') and interests.endswith(']'):
                        interests = eval(interests)
                    else:
                        interests = [i.strip() for i in interests.split(',')]
                except:
                    interests = [interests]
            all_interests.update(interests)
        
        all_interests = sorted(list(all_interests))
        interest_cols = [f"interest_{i}" for i in all_interests]
        
        for i in all_interests:
            df_clean[f'interest_{i}'] = df_clean['Interests'].apply(
                lambda x: 1 if i in (
                    eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']')
                    else [x.strip() for x in x.split(',')] if isinstance(x, str)
                    else x
                ) else 0
            )
        
        # 2. Process sites visited - create one-hot encoding
        all_sites = set()
        for sites in df_clean['Sites Visited']:
            if isinstance(sites, str):
                try:
                    if sites.startswith('[') and sites.endswith(']'):
                        sites = eval(sites)
                    else:
                        sites = [s.strip() for s in sites.split(',')]
                except:
                    sites = [sites]
            all_sites.update(sites)
        
        all_sites = sorted(list(all_sites))
        site_cols = [f"site_{s}" for s in all_sites]
        
        for s in all_sites:
            df_clean[f'site_{s}'] = df_clean['Sites Visited'].apply(
                lambda x: 1 if s in (
                    eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']')
                    else [s.strip() for s in x.split(',')] if isinstance(x, str)
                    else x
                ) else 0
            )
        
        # 3. Process accessibility (boolean to int)
        df_clean['Accessibility'] = df_clean['Accessibility'].astype(int)
        
        # 4. Scale numerical features
        scaler = StandardScaler()
        df_clean['Age_scaled'] = scaler.fit_transform(df_clean[['Age']])
        df_clean['Preferred Tour Duration_scaled'] = scaler.fit_transform(df_clean[['Preferred Tour Duration']])
        
        # 5. Normalize satisfaction scores to 0-1 range
        max_satisfaction = 5.0  # Assuming the maximum rating is 5
        df_clean['Satisfaction_normalized'] = df_clean['Satisfaction'] / max_satisfaction
        
        # Prepare feature matrix
        feature_cols = ['Age_scaled', 'Preferred Tour Duration_scaled', 'Accessibility'] + interest_cols + site_cols
        X = df_clean[feature_cols].values
        y = df_clean['Satisfaction'].values  # Use original satisfaction values
        
        return X, y, feature_cols
    
    def _create_feature_vector(self, age, interests, preferred_duration, accessibility, sites):
        """Create a feature vector for prediction."""
        # Create empty feature dict
        features = {}
        
        # Process numerical features
        features['Age_scaled'] = self.age_scaler.transform([[age]])[0][0]
        features['Preferred Tour Duration_scaled'] = self.duration_scaler.transform([[preferred_duration]])[0][0]
        features['Accessibility'] = int(accessibility)
        
        # Process interests
        for col in self.feature_columns:
            if col.startswith('interest_'):
                interest = col[len('interest_'):]
                features[col] = 1 if interest in interests else 0
        
        # Process sites
        for col in self.feature_columns:
            if col.startswith('site_'):
                site = col[len('site_'):]
                features[col] = 1 if site in sites else 0
        
        # Convert to feature vector
        feature_vector = [features.get(col, 0) for col in self.feature_columns]
        
        return feature_vector
    
    def predict_satisfaction(self, feature_vector):
        """Predict satisfaction using the Random Forest model."""
        prediction = self.model.predict([feature_vector])[0]
        return prediction
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        """
        Recommend sites based on Random Forest predictions.
        """
        # Get all candidate sites
        candidate_sites = self.all_sites
        
        # Calculate predicted satisfaction for each site
        site_satisfaction = {}
        for site in candidate_sites:
            # Create feature vector with just this site
            feature_vector = self._create_feature_vector(
                age, interests, preferred_duration, accessibility, [site]
            )
            
            # Predict satisfaction
            satisfaction = self.predict_satisfaction(feature_vector)
            site_satisfaction[site] = satisfaction
        
        # Calculate efficiency (satisfaction/duration)
        site_efficiency = {
            site: site_satisfaction[site] / self.site_durations.get(site, 1.0)
            for site in candidate_sites
        }
        
        # Sort by efficiency
        sorted_sites = sorted(candidate_sites, key=lambda s: site_efficiency[s], reverse=True)
        
        # Select sites greedily within duration constraint
        recommended_sites = []
        total_duration = 0
        
        for site in sorted_sites:
            site_duration = self.site_durations.get(site, 1.0)
            if total_duration + site_duration <= preferred_duration:
                recommended_sites.append(site)
                total_duration += site_duration
                
                if len(recommended_sites) >= top_k:
                    break
        
        # Calculate expected satisfaction for the combination
        if recommended_sites:
            feature_vector = self._create_feature_vector(
                age, interests, preferred_duration, accessibility, recommended_sites
            )
            expected_satisfaction = self.predict_satisfaction(feature_vector)
        else:
            expected_satisfaction = 0
        
        return {
            'recommended_sites': recommended_sites,
            'expected_satisfaction': expected_satisfaction,
            'total_duration': total_duration
        }


class MLPRecommender:
    """
    Uses the MLP model from TouristRecommendationSystem to predict satisfaction.
    """
    def __init__(self, train_df, **kwargs):
        random_state = kwargs.get('random_state', 42)
        torch.manual_seed(random_state)
        
        print("Training MLP recommendation model...")
        self.recommender = TouristRecommendationSystem()
        self.recommender.train_model(train_df, epochs=kwargs.get('epochs', 100))
        print("MLP model training complete.")
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        """
        Use the MLP model to recommend sites.
        """
        return self.recommender.recommend_sites(
            age, interests, preferred_duration, accessibility, top_k=top_k
        )


class HybridRecommender:
    """
    Combines multiple recommendation approaches.
    """
    def __init__(self, train_df, **kwargs):
        self.train_df = train_df
        random_state = kwargs.get('random_state', 42)
        
        # Initialize component recommenders
        print("Initializing hybrid recommendation components...")
        self.popularity_rec = PopularityRecommender(train_df)
        self.interest_rec = InterestMatchingRecommender(train_df)
        self.satisfaction_rec = SatisfactionMaximizingRecommender(train_df)
        print("Hybrid components initialized.")
        
        # Weight for each component (default equal weights)
        self.weights = kwargs.get('weights', {
            'popularity': 0.33,
            'interest': 0.33,
            'satisfaction': 0.34
        })
        
        # Extract site durations
        self.site_durations = {}
        self._extract_site_durations(train_df)
    
    def _extract_site_durations(self, df):
        """Extract average duration for each site."""
        site_durations = {}
        
        for _, row in df.iterrows():
            # Get sites
            if isinstance(row['Sites Visited'], str):
                if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                    sites = eval(row['Sites Visited'])
                else:
                    sites = [s.strip() for s in row['Sites Visited'].split(',')]
            else:
                sites = row['Sites Visited']
            
            # Calculate average duration per site
            if len(sites) > 0:
                avg_duration = row['Tour Duration'] / len(sites)
                for site in sites:
                    if site not in site_durations:
                        site_durations[site] = []
                    site_durations[site].append(avg_duration)
        
        # Calculate average duration for each site
        for site in site_durations:
            if site_durations[site]:
                self.site_durations[site] = sum(site_durations[site]) / len(site_durations[site])
            else:
                self.site_durations[site] = 1.0  # Default to 1 hour if unknown
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, top_k=2):
        """
        Recommend sites using a hybrid approach.
        """
        # Get recommendations from each component
        popularity_rec = self.popularity_rec.recommend_sites(
            age, interests, preferred_duration, accessibility, top_k=10
        )
        
        interest_rec = self.interest_rec.recommend_sites(
            age, interests, preferred_duration, accessibility, top_k=10
        )
        
        satisfaction_rec = self.satisfaction_rec.recommend_sites(
            age, interests, preferred_duration, accessibility, top_k=10
        )
        
        # Collect all candidate sites
        candidate_sites = set()
        candidate_sites.update(popularity_rec['recommended_sites'])
        candidate_sites.update(interest_rec['recommended_sites'])
        candidate_sites.update(satisfaction_rec['recommended_sites'])
        
        # Score each site based on its position in each recommendation list
        site_scores = {site: 0 for site in candidate_sites}
        
        # Add popularity score
        for i, site in enumerate(popularity_rec['recommended_sites']):
            site_scores[site] += self.weights['popularity'] * (10 - i) / 10
        
        # Add interest score
        for i, site in enumerate(interest_rec['recommended_sites']):
            site_scores[site] += self.weights['interest'] * (10 - i) / 10
        
        # Add satisfaction score
        for i, site in enumerate(satisfaction_rec['recommended_sites']):
            site_scores[site] += self.weights['satisfaction'] * (10 - i) / 10
        
        # Sort sites by combined score
        sorted_sites = sorted(site_scores.keys(), key=lambda s: site_scores[s], reverse=True)
        
        # Select sites greedily within duration constraint
        recommended_sites = []
        total_duration = 0
        
        for site in sorted_sites:
            site_duration = self.site_durations.get(site, 1.0)
            if total_duration + site_duration <= preferred_duration:
                recommended_sites.append(site)
                total_duration += site_duration
                
                if len(recommended_sites) >= top_k:
                    break
        
        # Estimate expected satisfaction (weighted average of component predictions)
        component_satisfactions = {
            'popularity': popularity_rec['expected_satisfaction'],
            'interest': interest_rec['expected_satisfaction'],
            'satisfaction': satisfaction_rec['expected_satisfaction']
        }
        
        # Calculate weighted average satisfaction
        expected_satisfaction = sum(
            component_satisfactions[key] * self.weights[key]
            for key in self.weights
        )
        
        return {
            'recommended_sites': recommended_sites,
            'expected_satisfaction': expected_satisfaction,
            'total_duration': total_duration
        }


def run_multiple_strategy_comparisons(num_runs=10, base_seed=42):
    """
    Run multiple evaluations of different recommendation strategies with different random seeds.
    
    Args:
        num_runs: Number of evaluation runs to perform
        base_seed: Base random seed to use (will be incremented for each run)
    
    Returns:
        Dictionary containing aggregated results across all runs
    """
    print(f"Starting evaluation with {num_runs} runs using test set data...")
    
    # Storage for results from all runs
    all_results = []
    all_statistics = {}
    all_best_strategies = []
    
    # Run evaluations with different random seeds
    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx
        print(f"\nRun {run_idx+1}/{num_runs} (seed={run_seed}):")
        
        # Initialize new evaluator with current seed
        evaluator = MultiStrategyEvaluator('tourism_dataset.csv', random_state=run_seed)
        
        # Add recommendation strategies with current random seed
        evaluator.add_strategy("MLP Model", MLPRecommender, epochs=100, random_state=run_seed)
        evaluator.add_strategy("Popularity-Based", PopularityRecommender)
        evaluator.add_strategy("Interest-Matching", InterestMatchingRecommender)
        evaluator.add_strategy("Satisfaction-Maximizing", SatisfactionMaximizingRecommender)
        evaluator.add_strategy("Least-Popular", LeastPopularRecommender, random_state=run_seed)
        evaluator.add_strategy("Random", RandomRecommender, random_state=run_seed)
        evaluator.add_strategy("Hybrid Approach", HybridRecommender, random_state=run_seed)
        
        # Run evaluation on the test set
        results = evaluator.evaluate_strategies_on_test_set(random_seed=run_seed)
        
        # Store results
        all_results.append(results)
        all_best_strategies.append(results['best_strategy'])
        
        # Initialize statistics storage on first run
        if run_idx == 0:
            for name in results['statistics']:
                all_statistics[name] = {
                    'satisfaction_across_runs': [],
                    'avg_satisfaction_across_runs': [],
                    'std_satisfaction_across_runs': [],
                    'mae_across_runs': [],
                    'rmse_across_runs': []
                }
        
        # Store statistics for each strategy
        for name, stats in results['statistics'].items():
            all_statistics[name]['satisfaction_across_runs'].append(stats['satisfaction_scores'])
            all_statistics[name]['avg_satisfaction_across_runs'].append(stats['avg_satisfaction'])
            all_statistics[name]['std_satisfaction_across_runs'].append(stats['std_satisfaction'])
            all_statistics[name]['mae_across_runs'].append(stats['mae'])
            all_statistics[name]['rmse_across_runs'].append(stats['rmse'])
    
    # Aggregate results across all runs
    
    # 1. Compute average statistics for each strategy
    avg_statistics = {}
    std_statistics = {}
    
    for name, stats in all_statistics.items():
        avg_satisfaction = np.mean(stats['avg_satisfaction_across_runs'])
        std_satisfaction = np.std(stats['avg_satisfaction_across_runs'])
        avg_mae = np.mean(stats['mae_across_runs'])
        std_mae = np.std(stats['mae_across_runs'])
        avg_rmse = np.mean(stats['rmse_across_runs'])
        std_rmse = np.std(stats['rmse_across_runs'])
        
        avg_statistics[name] = {
            'avg_satisfaction': avg_satisfaction,
            'median_satisfaction': np.median(stats['avg_satisfaction_across_runs']),
            'satisfaction_across_runs': stats['avg_satisfaction_across_runs'],
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse
        }
        
        std_statistics[name] = {
            'avg_satisfaction': std_satisfaction,
            'mae': std_mae,
            'rmse': std_rmse
        }
    
    # 2. Count occurrences of each strategy as the best
    best_strategy_counts = {}
    for strategy in all_best_strategies:
        if strategy not in best_strategy_counts:
            best_strategy_counts[strategy] = 0
        best_strategy_counts[strategy] += 1
    
    # 3. Aggregate pairwise test results
    avg_pairwise_tests = {}
    
    # Initialize with the first run's test pairs
    for test_name in all_results[0]['pairwise_tests']:
        avg_pairwise_tests[test_name] = {
            'p_values': [],
            'effect_sizes': [],
            'significant_count': 0,
            'better_counts': {}
        }
    
    # Collect data across all runs
    for run_results in all_results:
        for test_name, test_result in run_results['pairwise_tests'].items():
            # Track p-values and effect sizes
            avg_pairwise_tests[test_name]['p_values'].append(test_result['p_value'])
            avg_pairwise_tests[test_name]['effect_sizes'].append(test_result['effect_size'])
            
            # Count significant tests
            if test_result['significant']:
                avg_pairwise_tests[test_name]['significant_count'] += 1
            
            # Count which strategy was better
            better = test_result['better']
            if better not in avg_pairwise_tests[test_name]['better_counts']:
                avg_pairwise_tests[test_name]['better_counts'][better] = 0
            avg_pairwise_tests[test_name]['better_counts'][better] += 1
    
    # Calculate averages for pairwise tests
    for test_name, test_data in avg_pairwise_tests.items():
        test_data['p_value'] = np.mean(test_data['p_values'])
        test_data['effect_size'] = np.mean(test_data['effect_sizes'])
        
        # Determine which strategy was better most often
        better_strategy = max(test_data['better_counts'].items(), key=lambda x: x[1])[0]
        test_data['better'] = better_strategy
    
    # Package aggregated results
    aggregated_results = {
        'avg_statistics': avg_statistics,
        'std_statistics': std_statistics,
        'avg_pairwise_tests': avg_pairwise_tests,
        'best_strategy_counts': best_strategy_counts,
        'num_runs': num_runs
    }
    
    return aggregated_results


def run_strategy_comparison():
    """
    Run a comparison of different recommendation strategies.
    """
    # Run multiple strategy comparisons using the test set
    aggregated_results = run_multiple_strategy_comparisons(num_runs=10)
    
    # Create a new evaluator just for visualization
    evaluator = MultiStrategyEvaluator('tourism_dataset.csv')
    
    # Visualize aggregated results
    evaluator.visualize_results(aggregated_results)
    
    return aggregated_results


if __name__ == "__main__":
    run_strategy_comparison()