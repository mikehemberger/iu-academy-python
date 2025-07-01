import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from bokeh.plotting import figure, output_file, save
import os

# Custom exception
class CustomException(Exception):
    """
    Raised for domain-specific errors in function mapping.
    """
    pass

# Base class for loading and holding data
class BaseDataHandler:
    """
    Loads CSV files for training, ideal, and test data.
    Provides common data loading and validation functionality.
    """
    def __init__(self, train_path, ideal_path, test_path):
        self.train_path = train_path
        self.ideal_path = ideal_path
        self.test_path = test_path
        self.train_df = None
        self.ideal_df = None
        self.test_df = None

    def load_data(self):
        try:
            self.train_df = pd.read_csv(self.train_path)
            self.ideal_df = pd.read_csv(self.ideal_path)
            self.test_df = pd.read_csv(self.test_path)
        except FileNotFoundError as e:
            raise CustomException(f"File not found: {e}")
        except pd.errors.ParserError as e:
            raise CustomException(f"Error parsing CSV: {e}")

# SQLite DB handler
class SQLiteManager:
    """
    Manages creation and interaction with a SQLite database using SQLAlchemy.
    """
    def __init__(self, db_path=':memory:'):
        self.engine = create_engine(f'sqlite:///{db_path}')

    def store_dataframe(self, df, table_name):
        try:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        except SQLAlchemyError as e:
            raise CustomException(f"Database error: {e}")

    def read_table(self, table_name):
        try:
            return pd.read_sql_table(table_name, self.engine)
        except SQLAlchemyError as e:
            raise CustomException(f"Database read error: {e}")

# Selects the best matching ideal functions for each training function
class FunctionSelector(BaseDataHandler):
    """
    Selects ideal functions that best fit each training dataset
    by minimizing the sum of squared differences (least squares method).
    """
    def __init__(self, train_path, ideal_path, test_path):
        super().__init__(train_path, ideal_path, test_path)
        self.best_matches = {}  # Maps train column name to ideal column name

    def select_best_functions(self):
        self.load_data()
        train_x = self.train_df['x']
        ideal_x = self.ideal_df['x']

        if not train_x.equals(ideal_x):
            raise CustomException("x-values of training and ideal functions do not match")

        for train_col in self.train_df.columns[1:]:
            min_ssd = np.inf
            best_col = None
            for ideal_col in self.ideal_df.columns[1:]:
                ssd = np.sum((self.train_df[train_col] - self.ideal_df[ideal_col]) ** 2)
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_col = ideal_col
            self.best_matches[train_col] = best_col

        return self.best_matches

# Maps test data points to selected ideal functions within deviation threshold
class TestMapper(BaseDataHandler):
    """
    Maps test data points to ideal functions that stay within the allowed deviation threshold
    (max training deviation × sqrt(2)).
    """
    def __init__(self, train_path, ideal_path, test_path, best_matches):
        super().__init__(train_path, ideal_path, test_path)
        self.best_matches = best_matches
        self.mapping_results = pd.DataFrame(columns=['x', 'y', 'delta_y', 'ideal_func'])

    def map_test_points(self):
        self.load_data()
        thresholds = {}

        for train_col, ideal_col in self.best_matches.items():
            deviation = np.abs(self.train_df[train_col] - self.ideal_df[ideal_col])
            thresholds[ideal_col] = deviation.max() * np.sqrt(2)

        for _, row in self.test_df.iterrows():
            x, y = row['x'], row['y']
            if x in self.ideal_df['x'].values:
                idx = self.ideal_df[self.ideal_df['x'] == x].index[0]
                best_fit = None
                smallest_dev = np.inf

                for ideal_col, threshold in thresholds.items():
                    ideal_y = self.ideal_df.at[idx, ideal_col]
                    deviation = abs(y - ideal_y)
                    if deviation <= threshold and deviation < smallest_dev:
                        best_fit = ideal_col
                        smallest_dev = deviation

                if best_fit:
                    self.mapping_results.loc[len(self.mapping_results)] = [x, y, smallest_dev, best_fit]

        return self.mapping_results

# Visualizes data using Bokeh
class Visualizer:
    """
    Generates visualizations for training data, selected ideal functions,
    test data, and matched test points using Bokeh.
    """
    @staticmethod
    def plot_data(train_df, ideal_df, test_df, best_matches, mapping_results, output_path="plot.html"):
        p = figure(title="Training, Ideal, and Test Data", x_axis_label="x", y_axis_label="y", width=900, height=500)

        colors = ["red", "green", "blue", "orange"]

        for i, train_col in enumerate(train_df.columns[1:]):
            p.line(train_df['x'], train_df[train_col], legend_label=f"Train: {train_col}", line_color=colors[i], line_dash="dotted")

        for i, train_col in enumerate(train_df.columns[1:]):
            ideal_col = best_matches[train_col]
            p.line(ideal_df['x'], ideal_df[ideal_col], legend_label=f"Ideal: {ideal_col}", line_color=colors[i], line_width=2)

        p.circle(test_df['x'], test_df['y'], size=5, color="gray", legend_label="Test Points")

        p.circle(mapping_results['x'], mapping_results['y'], size=6, color="black", legend_label="Mapped Points", fill_alpha=0.6)

        p.legend.location = "top_left"
        output_file(output_path)
        save(p)

# Example usage
if __name__ == "__main__":
    # Note: Make sure .csv files are located in root directory or adjust paths accordingly
    TRAIN_PATH = "train.csv"
    IDEAL_PATH = "ideal.csv"
    TEST_PATH = "test.csv"
    DB_PATH = "output.db"

    # Step 1: Select best ideal functions
    selector = FunctionSelector(TRAIN_PATH, IDEAL_PATH, TEST_PATH)
    best_matches = selector.select_best_functions()

    # Step 2: Map test data to ideal functions
    mapper = TestMapper(TRAIN_PATH, IDEAL_PATH, TEST_PATH, best_matches)
    results = mapper.map_test_points()

    # Step 3: Save everything to SQLite
    db = SQLiteManager(DB_PATH)
    db.store_dataframe(selector.train_df, "train")
    db.store_dataframe(selector.ideal_df, "ideal")
    db.store_dataframe(mapper.test_df, "test")
    db.store_dataframe(results, "results")

    # Step 4: Visualize
    Visualizer.plot_data(selector.train_df, selector.ideal_df, mapper.test_df, best_matches, results)

    print("All steps completed. Output saved to SQLite and HTML plot.")
