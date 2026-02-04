import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
    
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


numeric_features = ['AA', 'AMPS', 'CMC', 'PAA', 'PVA', 'CBR', 'RT (℃)', 'Rti (h)', 'DT (℃)', 'UC (%)', 'Rtemp (℃)', 'pH-SR']
categorical_features = ['CRA', 'ULM']
target = 'K (h⁻¹)'


rf_params = {
    'n_estimators': 102,
    'max_depth': 16,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 0.3,
    'bootstrap': True,
    'criterion': 'squared_error',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

class HydrogelPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Release Rate Prediction Tool")
        self.root.geometry("800x700")

        
        self.model_pipeline = None
        self.current_data_path = None
        self.feature_columns = None
        self.input_widgets = {} 

        self.create_widgets()
        self.load_default_dataset() 

    def create_widgets(self):
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)

        
        control_frame = ttk.LabelFrame(main_frame, text="1. Model Training", padding="10")
        control_frame.pack(fill='x', pady=(0, 10))

        
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill='x', pady=5)
        ttk.Button(file_frame, text="Upload New Dataset (.csv)", command=self.upload_file).pack(side='left')
        self.data_status_label = ttk.Label(file_frame, text="Loading default dataset...")
        self.data_status_label.pack(side='left', padx=20)
        
        
        self.train_button = ttk.Button(control_frame, text="Train Model", command=self.train_model, state='disabled')
        self.train_button.pack(pady=5, anchor='w')
        
        self.model_status_label = ttk.Label(control_frame, text="Model Status: Not Trained", foreground="red")
        self.model_status_label.pack(pady=5, anchor='w')

        
        input_frame = ttk.LabelFrame(main_frame, text="2. Input Feature Values for Prediction", padding="10")
        input_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        
        canvas = tk.Canvas(input_frame)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
       
        result_frame = ttk.LabelFrame(main_frame, text="3. Prediction Result", padding="10")
        result_frame.pack(fill='x')

        predict_button_frame = ttk.Frame(result_frame)
        predict_button_frame.pack(fill='x')
        self.predict_button = ttk.Button(predict_button_frame, text="Predict", command=self.predict, state='disabled')
        self.predict_button.pack(side='left', pady=5)
        
        self.result_label = ttk.Label(result_frame, text="Predicted Value (K h⁻¹): --", font=("Helvetica", 16, "bold"))
        self.result_label.pack(pady=10)

    def load_default_dataset(self):
        """Attempt to load the default dataset"""
        default_path = resource_path(os.path.join('data', 'data.csv'))
        if os.path.exists(default_path):
            self.current_data_path = default_path
            self.data_status_label.config(text=f"Default dataset loaded: {os.path.basename(default_path)}")
            self.train_button.config(state='normal')
        else:
            self.data_status_label.config(text="Default dataset not found. Please upload a new dataset.", foreground="orange")
            messagebox.showwarning("Warning", f"Default dataset file not found:\n{default_path}\n\nPlease click the 'Upload New Dataset' button to begin.")

    def upload_file(self):
        """Open a file dialog to select a CSV file"""
        f_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if f_path:
            self.current_data_path = f_path
            self.data_status_label.config(text=f"File selected: {os.path.basename(f_path)}")
            self.train_button.config(state='normal')
            self.reset_model_and_inputs()

    def reset_model_and_inputs(self):
        """Reset model status and clear input fields"""
        self.model_pipeline = None
        self.model_status_label.config(text="Model Status: Not Trained", foreground="red")
        self.predict_button.config(state='disabled')
        self.result_label.config(text="Predicted Value (K h⁻¹): --")

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.input_widgets = {}

    def train_model(self):
        """Execute the core logic for data loading and model training"""
        if not self.current_data_path:
            messagebox.showerror("Error", "Please select a dataset first!")
            return

        try:
            # 1. Load Data
            df = pd.read_csv(self.current_data_path)
            
            # 2. Check for required columns
            required_cols = numeric_features + categorical_features + [target]
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                messagebox.showerror("Data Format Error", f"The dataset is missing required columns: {', '.join(missing_cols)}")
                return

            X = df[numeric_features + categorical_features]
            y = df[target]
            self.feature_columns = X.columns 

            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ],
                remainder='passthrough'
            )
            self.model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(**rf_params))
            ])

           
            self.model_pipeline.fit(X, y)
            
           
            self.model_status_label.config(text="Model Status: Trained Successfully!", foreground="green")
            self.predict_button.config(state='normal')
            
            
            self.create_input_fields(df)
            
            messagebox.showinfo("Success", "Model training complete! You can now input feature values below to make a prediction.")

        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during model training:\n{e}")

    def create_input_fields(self, df):
        """Dynamically create input controls based on features"""
        # Clear old input fields
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.input_widgets = {}

        row = 0
        
        for feature in numeric_features:
            ttk.Label(self.scrollable_frame, text=f"{feature}:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
            entry = ttk.Entry(self.scrollable_frame, width=20)
            entry.grid(row=row, column=1, padx=5, pady=2)
            self.input_widgets[feature] = entry
            row += 1
            
       
        for feature in categorical_features:
            ttk.Label(self.scrollable_frame, text=f"{feature}:").grid(row=row, column=0, sticky='w', padx=5, pady=2)
            unique_values = df[feature].unique().tolist()
            combobox = ttk.Combobox(self.scrollable_frame, values=unique_values, width=18)
            combobox.grid(row=row, column=1, padx=5, pady=2)
            combobox.current(0) 
            self.input_widgets[feature] = combobox
            row += 1

    def predict(self):
        """Collect input values and make a prediction"""
        if not self.model_pipeline:
            messagebox.showerror("Error", "Please train the model first!")
            return

        try:
          
            input_data = {}
            for feature, widget in self.input_widgets.items():
                value = widget.get()
                if not value:
                    messagebox.showerror("Input Error", f"Please fill in the value for '{feature}'.")
                    return
               
                if feature in numeric_features:
                    input_data[feature] = float(value)
                else:
                    input_data[feature] = value
            
           
            input_df = pd.DataFrame([input_data], columns=self.feature_columns)

         
            prediction = self.model_pipeline.predict(input_df)[0]
            
          
            self.result_label.config(text=f"Predicted Value (K h⁻¹): {prediction:.4f}")

        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check if the input values are correct.\nError details: {e}")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HydrogelPredictorApp(root)
    root.mainloop()
