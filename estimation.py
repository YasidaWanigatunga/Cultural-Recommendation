import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class TourismDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MLPSatisfactionPredictor(nn.Module):
    """Standard Multi-Layer Perceptron (MLP) for satisfaction prediction."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLPSatisfactionPredictor, self).__init__()
        
        # Define the MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class DNNSatisfactionPredictor(nn.Module):
    """Deep Neural Network (DNN) for satisfaction prediction."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3, activation='relu'):
        super(DNNSatisfactionPredictor, self).__init__()
        
        # Define the DNN layers
        layers = []
        
        # Input layer with batch normalization
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        
        # Activation selection
        if activation == 'relu':
            act_func = nn.ReLU()
        elif activation == 'leaky_relu':
            act_func = nn.LeakyReLU()
        else:
            act_func = nn.GELU()
        
        layers.append(act_func)
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers with increased complexity
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(act_func)
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class RNNSatisfactionPredictor(nn.Module):
    """Recurrent Neural Network (RNN) for satisfaction prediction."""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(RNNSatisfactionPredictor, self).__init__()
        
        # Handle case where hidden_size might be a list
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]
        
        # RNN layers with hardcoded num_layers=2
        self.rnn = nn.RNN(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            dropout=dropout_rate, 
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # Ensure input is 3D tensor (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        
        # RNN output
        out, _ = self.rnn(x)
        
        # Handle different output dimensions
        if out.dim() == 3:
            # Take the last time step
            out = out[:, -1, :]
        elif out.dim() == 2:
            # If already 2D, use as is
            pass
        else:
            raise ValueError(f"Unexpected output dimension: {out.dim()}")
        
        # Pass through fully connected layers
        return self.fc(out)

class LSTMSatisfactionPredictor(nn.Module):
    """Long Short-Term Memory (LSTM) for satisfaction prediction."""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(LSTMSatisfactionPredictor, self).__init__()
        
        # Handle case where hidden_size might be a list
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]
        
        # LSTM layers with hardcoded num_layers=2
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            dropout=dropout_rate, 
            batch_first=True
        )
        
        # Fully connected output layer with additional complexity
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # Ensure input is 3D tensor (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        
        # LSTM output
        out, _ = self.lstm(x)
        
        # Handle different output dimensions
        if out.dim() == 3:
            # Take the last time step
            out = out[:, -1, :]
        elif out.dim() == 2:
            # If already 2D, use as is
            pass
        else:
            raise ValueError(f"Unexpected output dimension: {out.dim()}")
        
        # Pass through fully connected layers
        return self.fc(out)

class GRUSatisfactionPredictor(nn.Module):
    """Gated Recurrent Unit (GRU) for satisfaction prediction."""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(GRUSatisfactionPredictor, self).__init__()
        
        # Handle case where hidden_size might be a list
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]
        
        # GRU layers with hardcoded num_layers=2
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=2, 
            dropout=dropout_rate, 
            batch_first=True
        )
        
        # Fully connected output layer with residual connections
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        # Ensure input is 3D tensor (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        
        # GRU output
        out, _ = self.gru(x)
        
        # Handle different output dimensions
        if out.dim() == 3:
            # Take the last time step
            out = out[:, -1, :]
        elif out.dim() == 2:
            # If already 2D, use as is
            pass
        else:
            raise ValueError(f"Unexpected output dimension: {out.dim()}")
        
        # Pass through fully connected layers
        return self.fc(out)

class TouristRecommendationSystem:
    def __init__(self):
        self.site_encoders = {}
        self.interest_encoders = {}
        self.age_scaler = StandardScaler()
        self.duration_scaler = StandardScaler()
        self.models = []  # Store multiple models from different initializations
        
        # Keep track of all possible sites and associated durations
        self.all_sites = []
        self.site_durations = {}
        
        # Track performance metrics across initializations
        self.performance_metrics = {
            'train_losses': [],
            'test_losses': [],
            'final_mses': []
        }
    
    def preprocess_data(self, df):
        """
        Preprocess the tourism dataset
        """
        # Clean up data
        df_clean = df.copy()
        
        # Handle categorical data and lists
        
        # 1. Process interests - create one-hot encoding
        all_interests = set()
        for interests in df_clean['Interests']:
            # Convert string representation to list if needed
            if isinstance(interests, str):
                try:
                    # Try to safely evaluate the string as a list
                    if interests.startswith('[') and interests.endswith(']'):
                        interests = eval(interests)
                    else:
                        # Handle comma-separated format without brackets
                        interests = [i.strip() for i in interests.split(',')]
                except:
                    # Fallback: treat as a single item
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
            # Convert string representation to list if needed
            if isinstance(sites, str):
                try:
                    # Try to safely evaluate the string as a list
                    if sites.startswith('[') and sites.startswith(']'):
                        sites = eval(sites)
                    else:
                        # Handle comma-separated format without brackets
                        sites = [s.strip() for s in sites.split(',')]
                except:
                    # Fallback: treat as a single item
                    sites = [sites]
            all_sites.update(sites)
        
        self.all_sites = sorted(list(all_sites))
        site_cols = [f"site_{s}" for s in self.all_sites]
        
        for s in self.all_sites:
            df_clean[f'site_{s}'] = df_clean['Sites Visited'].apply(
                lambda x: 1 if s in (
                    eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']')
                    else [s.strip() for s in x.split(',')] if isinstance(x, str)
                    else x
                ) else 0
            )
        
        # Calculate average duration per site
        site_durations = {}
        for idx, row in df_clean.iterrows():
            try:
                # Handle different formats of Sites Visited
                if isinstance(row['Sites Visited'], str):
                    if row['Sites Visited'].startswith('[') and row['Sites Visited'].endswith(']'):
                        sites = eval(row['Sites Visited'])
                    else:
                        sites = [s.strip() for s in row['Sites Visited'].split(',')]
                else:
                    sites = row['Sites Visited']
                
                if len(sites) > 0:
                    avg_duration = row['Tour Duration'] / len(sites)
                    for site in sites:
                        if site not in site_durations:
                            site_durations[site] = []
                        site_durations[site].append(avg_duration)
            except Exception as e:
                print(f"Warning: Error processing row {idx}: {e}")
        
        # Calculate average duration for each site
        for site in site_durations:
            self.site_durations[site] = sum(site_durations[site]) / len(site_durations[site])
        
        # 3. Process accessibility (boolean to int)
        df_clean['Accessibility'] = df_clean['Accessibility'].astype(int)
        
        # 4. Scale numerical features
        self.age_scaler.fit(df_clean[['Age']])
        self.duration_scaler.fit(df_clean[['Preferred Tour Duration']])
        
        df_clean['Age_scaled'] = self.age_scaler.transform(df_clean[['Age']])
        df_clean['Preferred Tour Duration_scaled'] = self.duration_scaler.transform(df_clean[['Preferred Tour Duration']])
        
        # 5. Normalize satisfaction scores to 0-1 range
        # Store the maximum satisfaction value for later use in prediction
        self.max_satisfaction = 5.0  # Assuming the maximum rating is 5
        # Normalize the satisfaction scores
        df_clean['Satisfaction_normalized'] = df_clean['Satisfaction'] / self.max_satisfaction
        
        # Prepare feature matrix
        feature_cols = ['Age_scaled', 'Preferred Tour Duration_scaled', 'Accessibility'] + interest_cols + site_cols
        X = df_clean[feature_cols].values
        y = df_clean['Satisfaction_normalized'].values  # Use normalized satisfaction
        
        return X, y, feature_cols
    
    def train_single_model(self, X_train, y_train, X_test, y_test, hidden_sizes, lr, batch_size, epochs, seed):
        """
        Train a single model with a specific random seed
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create datasets
        train_dataset = TourismDataset(X_train, y_train.reshape(-1, 1))
        test_dataset = TourismDataset(X_test, y_test.reshape(-1, 1))
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        input_size = X_train.shape[1]
        model = DNNSatisfactionPredictor(input_size, hidden_sizes, 1)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            epoch_train_loss = running_loss/len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    test_loss += criterion(outputs, targets).item()
            
            epoch_test_loss = test_loss/len(test_loader)
            test_losses.append(epoch_test_loss)
            
            if epoch % 10 == 0:
                print(f'Seed {seed}, Epoch {epoch}, Train Loss: {epoch_train_loss:.4f}, '
                      f'Test Loss: {epoch_test_loss:.4f}')
        
        # Final evaluation
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                predictions.extend(outputs.numpy().flatten())
                actuals.extend(targets.numpy().flatten())
        
        mse = np.mean((np.array(predictions) - np.array(actuals))**2)
        print(f'Seed {seed}, Final MSE: {mse:.4f}')
        
        return model, mse, train_losses, test_losses
    
    def train_model(self, df, hidden_sizes=[128, 64, 32], lr=0.001, batch_size=32, epochs=100, num_initializations=10, base_seed=42):
        """
        Train multiple models with different random initializations
        """
        # Clear previous models and metrics
        self.models = []
        self.performance_metrics = {
            'train_losses': [],
            'test_losses': [],
            'final_mses': []
        }
        
        print(f"Starting training with {num_initializations} random initializations...")
        
        # Preprocess data once
        X, y, self.feature_cols = self.preprocess_data(df)
        
        # Split data once
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        
        # Train models with different initializations
        for i in range(num_initializations):
            seed = base_seed + i
            print(f"\nTraining model {i+1}/{num_initializations} with seed {seed}...")
            
            model, mse, train_losses, test_losses = self.train_single_model(
                X_train, y_train, X_test, y_test, 
                hidden_sizes, lr, batch_size, epochs, seed
            )
            
            # Store the model and its performance metrics
            self.models.append(model)
            self.performance_metrics['final_mses'].append(mse)
            self.performance_metrics['train_losses'].append(train_losses)
            self.performance_metrics['test_losses'].append(test_losses)
        
        # Calculate and print average performance
        avg_mse = np.mean(self.performance_metrics['final_mses'])
        std_mse = np.std(self.performance_metrics['final_mses'])
        
        print("\n" + "="*50)
        print(f"Training complete for {num_initializations} random initializations")
        print(f"Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
        print(f"MSE Range: {min(self.performance_metrics['final_mses']):.4f} - {max(self.performance_metrics['final_mses']):.4f}")
        print("="*50)
        
        # Set the best model as the default one for predictions
        best_model_idx = np.argmin(self.performance_metrics['final_mses'])
        self.model = self.models[best_model_idx]
        print(f"Using the best model (seed {base_seed + best_model_idx}) with MSE: {self.performance_metrics['final_mses'][best_model_idx]:.4f}")
        
        return {
            'avg_mse': avg_mse,
            'std_mse': std_mse,
            'all_mses': self.performance_metrics['final_mses'],
            'best_model_idx': best_model_idx
        }
    
    def predict_satisfaction(self, age, interests, preferred_duration, accessibility, sites, model_idx=None):
        """
        Predict satisfaction for a given tourist and itinerary
        
        Args:
            age: Tourist's age
            interests: List of tourist's interests
            preferred_duration: Preferred tour duration
            accessibility: Accessibility requirement (boolean)
            sites: List of sites in the itinerary
            model_idx: Index of the model to use (None for the best model)
        """
        if len(self.models) == 0:
            raise ValueError("No models trained yet. Call train_model first.")
        
        # Select the model to use
        if model_idx is not None:
            if model_idx < 0 or model_idx >= len(self.models):
                raise ValueError(f"Invalid model index. Should be between 0 and {len(self.models)-1}")
            model = self.models[model_idx]
        else:
            # Use the best model by default
            model = self.model
        
        # Create feature vector
        features = {}
        
        # Process numerical features
        features['Age_scaled'] = self.age_scaler.transform([[age]])[0][0]
        features['Preferred Tour Duration_scaled'] = self.duration_scaler.transform([[preferred_duration]])[0][0]
        features['Accessibility'] = int(accessibility)
        
        # Process interests
        for i in sorted(list(self.interest_encoders.keys())):
            features[f'interest_{i}'] = 1 if i in interests else 0
        
        # Process sites
        for s in self.all_sites:
            features[f'site_{s}'] = 1 if s in sites else 0
        
        # Convert to feature vector
        feature_vector = [features[col] if col in features else 0 for col in self.feature_cols]
        
        # Predict
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor([feature_vector], dtype=torch.float32))
        
        return prediction.item()
    
    def predict_satisfaction_ensemble(self, age, interests, preferred_duration, accessibility, sites):
        """
        Predict satisfaction using the ensemble of all trained models
        
        Returns:
            avg_satisfaction: Average predicted satisfaction
            std_satisfaction: Standard deviation of predicted satisfaction
        """
        if len(self.models) == 0:
            raise ValueError("No models trained yet. Call train_model first.")
        
        # Get predictions from all models
        predictions = []
        for i in range(len(self.models)):
            pred = self.predict_satisfaction(age, interests, preferred_duration, accessibility, sites, model_idx=i)
            predictions.append(pred)
        
        # Calculate average and standard deviation
        avg_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return avg_pred, std_pred
    
    def recommend_sites(self, age, interests, preferred_duration, accessibility, candidate_sites=None, top_k=3, use_ensemble=True):
        """
        Recommend the optimal combination of sites that maximizes satisfaction
        within the given tour duration constraint
        
        Args:
            age: Tourist's age
            interests: List of tourist's interests
            preferred_duration: Preferred tour duration
            accessibility: Accessibility requirement (boolean)
            candidate_sites: List of candidate sites to consider (None for all sites)
            top_k: Maximum number of sites to recommend
            use_ensemble: Whether to use ensemble prediction (average of all models)
        """
        if len(self.models) == 0:
            raise ValueError("No models trained yet. Call train_model first.")
        
        # If no candidate sites provided, use all sites
        if candidate_sites is None:
            candidate_sites = self.all_sites
        
        # Step 1: Calculate predicted satisfaction for each site
        site_satisfaction = {}
        site_satisfaction_std = {}  # For ensemble predictions
        
        for site in candidate_sites:
            # Predict satisfaction for this site alone
            if use_ensemble:
                pred, pred_std = self.predict_satisfaction_ensemble(
                    age, interests, preferred_duration, 
                    accessibility, [site]
                )
                site_satisfaction[site] = pred
                site_satisfaction_std[site] = pred_std
            else:
                pred = self.predict_satisfaction(
                    age, interests, preferred_duration, 
                    accessibility, [site]
                )
                site_satisfaction[site] = pred
        
        # Step 2: Sort sites by satisfaction/duration ratio (greedy approach)
        site_efficiency = {
            site: site_satisfaction[site] / self.site_durations.get(site, 1.0)
            for site in candidate_sites
        }
        
        sorted_sites = sorted(
            candidate_sites, 
            key=lambda s: site_efficiency[s], 
            reverse=True
        )
        
        # Step 3: Select sites greedily until duration constraint is met
        selected_sites = []
        current_duration = 0
        
        for site in sorted_sites:
            site_duration = self.site_durations.get(site, 1.0)
            if current_duration + site_duration <= preferred_duration:
                selected_sites.append(site)
                current_duration += site_duration
                
                # Break if we've selected enough sites
                if len(selected_sites) >= top_k:
                    break
        
        # Step 4: Calculate total satisfaction
        if use_ensemble:
            total_satisfaction, satisfaction_std = self.predict_satisfaction_ensemble(
                age, interests, preferred_duration, 
                accessibility, selected_sites
            )
        else:
            total_satisfaction = self.predict_satisfaction(
                age, interests, preferred_duration, 
                accessibility, selected_sites
            )
            satisfaction_std = 0  # No std if not using ensemble
        
        # Convert normalized satisfaction back to original scale for user display
        unnormalized_satisfaction = total_satisfaction * self.max_satisfaction
        unnormalized_std = satisfaction_std * self.max_satisfaction if satisfaction_std else 0
        
        result = {
            'recommended_sites': selected_sites,
            'expected_satisfaction': unnormalized_satisfaction,
            'normalized_satisfaction': total_satisfaction,
            'total_duration': current_duration
        }
        
        # Add uncertainty estimates if using ensemble
        if use_ensemble:
            result['satisfaction_std'] = unnormalized_std
            result['satisfaction_confidence_interval'] = [
                max(0, unnormalized_satisfaction - 2 * unnormalized_std),
                min(self.max_satisfaction, unnormalized_satisfaction + 2 * unnormalized_std)
            ]
        
        return result

# Example usage
if __name__ == "__main__":
    # Load the dataset from the CSV file
    try:
        print("Loading tourism dataset...")
        df = pd.read_csv('tourism_dataset.csv')
        print(f"Successfully loaded dataset with {len(df)} records")
        
        # Display data info
        print("\nDataset overview:")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample interests: {df['Interests'].iloc[0]}")
        print(f"Sample sites visited: {df['Sites Visited'].iloc[0]}")
        print(f"Satisfaction range: {df['Satisfaction'].min()} - {df['Satisfaction'].max()}")
        
        # Initialize and train the recommendation system with multiple initializations
        print("\nTraining recommendation model with multiple initializations...")
        recommender = TouristRecommendationSystem()
        results = recommender.train_model(df, epochs=100, num_initializations=10)
        
        print("\nEnsemble model performance:")
        print(f"Average MSE: {results['avg_mse']:.4f} ± {results['std_mse']:.4f}")
        print(f"Individual MSEs: {[f'{mse:.4f}' for mse in results['all_mses']]}")
        
        # Make a recommendation for a new tourist using the ensemble
        test_tourist = {
            'age': 35,
            'interests': ['History', 'Art', 'Food'],
            'preferred_duration': 6.0,
            'accessibility': True
        }
        
        print("\nGenerating recommendations using ensemble prediction...")
        recommendations = recommender.recommend_sites(
            test_tourist['age'],
            test_tourist['interests'],
            test_tourist['preferred_duration'],
            test_tourist['accessibility'],
            use_ensemble=True
        )
        
        print(f"Recommended sites: {recommendations['recommended_sites']}")
        print(f"Expected satisfaction: {recommendations['expected_satisfaction']:.2f} ± {recommendations.get('satisfaction_std', 0):.2f}")
        if 'satisfaction_confidence_interval' in recommendations:
            print(f"95% Confidence interval: {[f'{ci:.2f}' for ci in recommendations['satisfaction_confidence_interval']]}")
        print(f"Total duration: {recommendations['total_duration']:.1f} hours")
        
    except FileNotFoundError:
        print("Error: Could not find 'tourism_dataset.csv'. Make sure the file is in the same directory as this script.")
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")